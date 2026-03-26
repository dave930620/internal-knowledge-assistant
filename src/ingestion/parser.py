# =============================================================================
# src/ingestion/parser.py
#
# PURPOSE:
#   Parses a single raw HTML file into structured content: plain text and a
#   list of sections (each section corresponds to a heading h1/h2/h3 and the
#   body text that follows it). Used by build_documents.py during ingestion.
#
# INPUT:
#   file_path (str or Path): path to a raw HTML file, e.g.
#     data/raw/engineering_blogs/aws/eng_aws_001.html
#
# OUTPUT:
#   A Python dict with the following shape:
#     {
#       "title":    str,           # extracted from <title> tag
#       "raw_text": str,           # full visible text of the page
#       "sections": [              # list of heading-delimited sections
#         {
#           "section_id":    "sec_0",
#           "section_title": str,
#           "section_level": int,  # 1, 2, or 3
#           "content":       str
#         }, ...
#       ]
#     }
# =============================================================================

from bs4 import BeautifulSoup
import uuid


def parse_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # 👉 remove script/style
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    # 👉 title
    title = soup.title.string if soup.title else "No Title"

    # 👉 extract text
    text = soup.get_text(separator="\n")

    # 👉 extract sections (簡單版)
    sections = []
    for i, tag in enumerate(soup.find_all(["h1", "h2", "h3"])):
        section_text = []

        for sib in tag.next_siblings:
            if sib.name in ["h1", "h2", "h3"]:
                break
            if hasattr(sib, "get_text"):
                section_text.append(sib.get_text())

        sections.append({
            "section_id": f"sec_{i}",
            "section_title": tag.get_text(),
            "section_level": int(tag.name[1]),
            "content": "\n".join(section_text)
        })

    return {
        "title": title,
        "raw_text": text,
        "sections": sections
    }