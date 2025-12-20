import arxiv
import pandas as pd
import os

# 1. Define the Query
# Focus: RAG, Agents, Low-resource languages (relevant to your FYP), and PEFT
query = (
    'abs:"retrieval augmented generation" OR '
    'abs:"agentic ai" OR '
    'abs:"low resource languages" OR '
    'abs:"parameter efficient fine tuning" OR '
    'abs:"multilingual llm"'
)

# 2. Setup the Client
client = arxiv.Client()
search = arxiv.Search(
    query=query,
    max_results=20,  # Get 40 to ensure we hit the 35 valid PDF target
    sort_by=arxiv.SortCriterion.Relevance
)

results = []
download_dir = "nlp_research_corpus"
os.makedirs(download_dir, exist_ok=True)

print(f"🚀 Starting search for papers on: LLMs, RAG, and Low-Resource NLP...\n")

# 3. Download Loop
count = 0
for result in client.results(search):
    if not result.pdf_url:
        continue
    
    # Clean filename to avoid OS errors
    safe_title = "".join([c for c in result.title if c.isalnum() or c in (' ', '-', '_')]).rstrip()
    filename = f"{safe_title[:50]}.pdf"
    file_path = os.path.join(download_dir, filename)
    
    # Metadata for the assignment deliverables
    paper_info = {
        "Title": result.title,
        "Authors": ", ".join([a.name for a in result.authors]),
        "Year": result.published.year,
        "Abstract": result.summary.replace("\n", " "), # Clean up newlines for CSV
        "DOI/URL": result.pdf_url,
        "Filename": filename,
        "Source": "ArXiv (Programmatic)" # To track source as requested
    }
    
    try:
        # Download the PDF
        result.download_pdf(dirpath=download_dir, filename=filename)
        results.append(paper_info)
        count += 1
        print(f"[{count}] Downloaded: {result.title[:60]}...")
    except Exception as e:
        print(f"❌ Failed to download {result.title}: {e}")

# 4. Save Metadata (Deliverable 1 Requirement)
df = pd.DataFrame(results)
df.to_csv("metadata.csv", index=False)
df.to_json("metadata.json", orient="records", indent=4) # Saving JSON too just in case

print(f"\n✅ COMPLETE: {len(df)} papers downloaded to '/{download_dir}'")
print(f"✅ Metadata saved to 'metadata.csv' and 'metadata.json'")