import fitz as pymupdf

class PDFLoader:
    def __init__(self, filePath: str):
        self.pdf_filePath = filePath
        self.doc = pymupdf.open(self.pdf_filePath)
        self.images = []
        self.text = []
        for i in range(len(self.doc)):
            # finding out the images and then appending them to an array
            self.images = self.doc[i].get_images()

            # finding the text part of the pdf
            self.text = self.doc[i].get_text()
        
    def __len__(self):
        return len(self.doc)

    def img_to_pix(self):
        print("hello")
        for imgIdx, img in enumerate(self.images):
            xref = img[0]
            print(img)
            pix = pymupdf.Pixmap(self.doc, xref)

            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            
            yield imgIdx, pix
            pix = None
        self.doc.close()
        print("finished generating the images for the pdf pages")