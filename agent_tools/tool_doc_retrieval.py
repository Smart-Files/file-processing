from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.pdf import BasePDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain.tools.retriever import create_retriever_tool


import os

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_documents_db(directory: str, persist_dir: str = "db"):
    documents = []


    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    try:
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        if vectordb: 
            print("Found Database: Importing!")
            return vectordb
    except Exception as e:
        print("Could not load DB from disk: ", e)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        loader = TextLoader(file_path)
        if filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        elif filename.endswith(".pdf"):
            loader = BasePDFLoader(file_path)
        elif filename.endswith(".html"):    
            loader = BSHTMLLoader(file_path)

        document = loader.load()
        documents.extend(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=PERSIST_DIR)

    return vectordb


def create_file_retrieval_tool():
    db = load_documents_db("llm_docs")
    retriever = db.as_retriever()

    file_tool = create_retriever_tool(
        retriever,
        "search_file_tools_docs",
        """Searches and returns excerpts from documentation for CLI usage of the following tools:
        - pandoc
            - Pandoc is a Haskell library for converting from one markup format to another, and a command-line tool that uses this library. It can read markdown and (subsets of) reStructuredText, HTML, and LaTeX, and it can write markdown, reStructuredText, HTML, LaTeX, ConTeXt, PDF, RTF, DocBook XML, OpenDocument XML, ODT, Word docx, GNU Texinfo, MediaWiki markup, EPUB, FictionBook2, Textile, groff, etc.
        - ffmpeg
            - FFmpeg is a free and open-source project consisting of a large software suite of libraries and programs for handling video, audio, and other multimedia files and streams.  FFmpeg is designed for command-line-based processing of video and audio files and is used for format transcoding, basic editing (trimming and concatenation), video scaling, video post-production effects, and standards compliance (SMPTE, ITU).
        - image magick
            - ImageMagick is a CLI tool for displaying, converting, and editing raster image and vector image files. It can read and write over 200 image file formats. ImageMagick can resize, flip, mirror, rotate, distort, shear and transform images, adjust image colors, apply various special effects, or draw text, lines, polygons, ellipses, and Bézier curves.
        - poppler
            - Poppler is a library for rendering PDF documents. It includes the following tools:
            - pdfimages saves images from a PDF file as PPM, PBM, PNG, TIFF, JPEG, JPEG2000, or JBIG2 files.
            - pdftotext converts PDF files to plain text.
            - pdftocairo renders PDF files as PNG, PS, EPS, SVG, or PDF files using cairo.
            - pdftohtml converts PDF files to HTML.
            - pdfunite merges several PDF files into one PDF file.

        These tools will help you to operations with files.

        Search any query with the tool name to get the documentation for that tool. For example, "pandoc convert markdown to pdf" will return the documentation for the pandoc tool.
        """,
    )

    return file_tool