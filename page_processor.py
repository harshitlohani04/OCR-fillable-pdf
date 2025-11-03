import cv2 as cv
from fitz import Pixmap
import numpy as np
import easyocr

# This will be responsible for processing everything in the pdfs image
class PageProcessor:
    reader = None
    def __init__(self, image: Pixmap):
        self.pix_img = image
        self._reader = self._reader_fetcher()
        self._threshold_y = 50
        self._threshold_x = 250
        self.detected_data = None

    def convert_pix_to_image(self):
        try:
            np_img = np.frombuffer(self.pix_img.samples, dtype=np.uint8).reshape(
                self.pix_img.height, self.pix_img.width, self.pix_img.n
            )
        except AttributeError:
            # Handle cases where 'image' might not be a Pixmap
            print("Error: Input is not a valid fitz.Pixmap object.")
            return []
        
        if np_img.shape[2] == 3:  # RGB
            np_img = cv.cvtColor(np_img, cv.COLOR_RGB2BGR)
        elif np_img.shape[2] == 4:  # RGBA
            np_img = cv.cvtColor(np_img, cv.COLOR_RGBA2BGRA)

        return np_img
    
    @classmethod
    def _reader_fetcher(cls):
        if not cls.reader:
            print("Loading the OCR model")
            cls.reader = easyocr.Reader(["en"], gpu=False)
        return cls.reader
    
    def extract_data_from_image(self):
        np_img = self.convert_pix_to_image()

        ocr_results = self._reader.readtext(np_img, width_ths=0.75, height_ths = 0.65)
        idToText_mapping = {}
        idToTextCoords_mapping = {}
        # ([[664, 222], [925, 222], [925, 302], [664, 302]],'Station',0.3260) --> Sample output from docs

        def standardize_coords(coords):
            x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0
            for x, y in coords:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            return [x_min, y_min, x_max, y_max]

        for idx, data in enumerate(ocr_results, start=1):
            idToText_mapping[idx] = data[1]
            idToTextCoords_mapping[idx] = standardize_coords(data[0])

        return (idToText_mapping, idToTextCoords_mapping)
    
    def remove_overlapping_lines(boxes, lines):
        overlapping_idxs = []
        for i, l_coords in enumerate(lines):
            for b_coords in boxes:
                xmindiff, ymindiff, xmaxdiff, ymaxdiff = b_coords[0]-l_coords[0], b_coords[1]-l_coords[1], b_coords[2]-l_coords[2], b_coords[3]-l_coords[3]
                if -5 < xmindiff < 5 and -5 < ymindiff < 5 and -5 < xmaxdiff < 5 and -5 < ymaxdiff < 5:
                    overlapping_idxs.append(i)

        for i in overlapping_idxs:
            del lines[i]

        return lines

    def find_fieldContours(self):
        image = self.convert_pix_to_image()
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (3, 3), 0)

        detected_fields = []
        detected_lines = []

        _, thresh = cv.threshold(blurred, 150, 255, cv.THRESH_BINARY_INV)

        contours, _ = cv.findContours(thresh, cv.RETR_EXTRENAL, cv.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            if cv.contourArea(cont)<100:
                continue
            perimeter = cv.arcLength(cont, True)
            epsilon = 0.04 * perimeter
            approx = cv.approxPolyDP(cont, epsilon, True)

            if len(approx) == 4:
                (x, y, w, h) = cv.boundingRect(approx)
                aspect_ratio = w/float(h)

                if aspect_ratio > 1.5:
                    detected_fields.append([x, y, x+w, y+h])

        edges = cv.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv.HoughLinesP(
            edges,
            rho = 1,
            theta = np.pi/180,
            threshold = 50,
            miniLineLength = 50,
            maxLineGap = 10
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1-y2) < 5: # thresholding for straight line detection
                    detected_lines.append([min(x1, x2), min(y1, y2), max(x2, x1), max(y2, y1)])

        final_lines = PageProcessor.remove_overlapping_lines(detected_fields, detected_lines)

        return {"boxes": detected_fields, "lines": final_lines}
    
    def is_inCenter(bbox, text) -> bool:
        x_center = (text[0]+text[2])/2
        y_center = (text[1] + text[3])/2

        if bbox[0]<=x_center<=bbox[2] and bbox[1]<=y_center<=bbox[3]:
            return True
        return False
    
    def is_validLine(text_coords, line_coords) -> bool:
        text_coords_xmin, t_ymin, text_coords_xmax, t_ymax = text_coords[0], text_coords[1], text_coords[2], text_coords[3]
        line_coords_xmin, l_ymin, line_coords_xmax, l_ymax = line_coords[0], line_coords[1], line_coords[2], line_coords[3]

        is_directlyUnder = abs(t_ymax - l_ymin) < 20

        overlap_start = max(text_coords_xmin, line_coords_xmin)
        overlap_end = min(text_coords_xmax, line_coords_xmax)

        overlap_width = overlap_end - overlap_start

        text_width = text_coords_xmax-text_coords_xmin
        text_width = text_width if text_width>0 else 1

        overlap = overlap_width/text_width > 0.5
        return True if is_directlyUnder and overlap else False
    
    def finalContoursMapping(self):
        contours = self.find_fieldContours()
        boxes = contours["boxes"]
        lines = boxes["lines"]
        contourToTextMapping = {}
        coveredIds = list()
        content, text = self.extract_data_from_image()
        for textid, coords in text.items():
            _, text_ymin, text_xmax, text_ymax = coords
            foundBox, foundLine = None, None
            for box in boxes:
                box_xmin, box_ymin, _, box_ymax = box
                if abs(box_ymin-text_ymin) < 10 and abs(box_ymax-text_ymax) < 10 and box_xmin - text_xmax > 20 and box not in contourToTextMapping.items():
                    if textid not in coveredIds:
                        foundBox = box
                        break

            for line in lines:
                line_xmin, line_ymin,  _, line_ymax = line
                if abs(line_ymin-text_ymin) < 10 and abs(line_ymax-text_ymax) < 10 and line_xmin - text_xmax > 20 and line not in contourToTextMapping.items():
                    if textid not in coveredIds:
                        foundLine = line
                        break
            if foundBox or foundLine:
                finalSelected = box if box else line
                contourToTextMapping[finalSelected] = content[textid]

        return contourToTextMapping
            


                
