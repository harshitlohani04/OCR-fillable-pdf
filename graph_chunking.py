from typing import List, Optional
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from networkx.algorithms import community

def semantic_graph_chunking(
        text: str,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_sentences_per_chunk: int = 1,
    ) -> List[str]:
    """
    Chunk text into semantically coherent sections using graph-based clustering.

    Args:
        text (str): Input document as a string.
        model_name (str): SentenceTransformer model name for embeddings.
        similarity_threshold (float): Minimum cosine similarity between sentences 
                                      to form an edge in the graph (0–1).
        min_sentences_per_chunk (int): Minimum number of sentences per semantic chunk.
        verbose (bool): Whether to print debug information.

    Returns:
        List[str]: A list of semantically meaningful text chunks.
    """

    # split text into sentences
    sentences = sent_tokenize(text)

    if len(sentences) == 0:
        return []

    # generate embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)

    # build similarity graph
    similarity_matrix = cosine_similarity(embeddings)
    G = nx.Graph()
    for i, sent in enumerate(sentences):
        G.add_node(i, text=sent)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > similarity_threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])

    # detect communities (clusters)
    communities = community.greedy_modularity_communities(G)

    # combine sentences within each cluster
    semantic_chunks = []
    for cluster in communities:
        if len(cluster) >= min_sentences_per_chunk:
            chunk = " ".join(G.nodes[i]["text"] for i in sorted(cluster))
            semantic_chunks.append(chunk)

    return semantic_chunks
