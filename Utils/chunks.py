import json
from pathlib import Path

def count_chunks_in_json_files(folder_path):
    """
    Count chunks in each JSON file in the specified folder.
    Expects JSON files with the format: list of document objects, each containing 'chunks' array.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Get all JSON files in the folder
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return
    
    print("=" * 80)
    print(f"CHUNK COUNT ANALYSIS - Folder: {folder_path}")
    print("=" * 80)
    print()
    
    total_chunks_all_files = 0
    total_documents_all_files = 0
    file_stats = []
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single document and list of documents
            if isinstance(data, list):
                documents = data
            else:
                documents = [data]
            
            total_chunks = 0
            total_docs = len(documents)
            
            # Count chunks per document within the file
            for doc in documents:
                if isinstance(doc, dict) and 'chunks' in doc:
                    total_chunks += len(doc['chunks'])
                elif isinstance(doc, dict) and 'original_document' in doc and 'chunks' in doc:
                    # Some formats have original_document wrapper
                    total_chunks += len(doc.get('chunks', []))
            
            file_stats.append({
                'filename': json_file.name,
                'documents': total_docs,
                'chunks': total_chunks
            })
            
            total_chunks_all_files += total_chunks
            total_documents_all_files += total_docs
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing {json_file.name}: {e}")
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
    
    # Display results
    print("📊 FILE-BY-FILE BREAKDOWN:")
    print("-" * 80)
    print(f"{'Filename':<40} {'Documents':<12} {'Chunks':<12}")
    print("-" * 80)
    
    for stat in sorted(file_stats, key=lambda x: x['filename']):
        print(f"{stat['filename']:<40} {stat['documents']:<12} {stat['chunks']:<12}")
    
    print("-" * 80)
    print()
    
    # Summary
    print("📈 SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"Total JSON files processed:  {len(json_files)}")
    print(f"Total documents across all files:  {total_documents_all_files}")
    print(f"Total chunks across all files:     {total_chunks_all_files}")
    print(f"Average chunks per file:           {total_chunks_all_files / len(json_files):.1f}")
    print(f"Average documents per file:        {total_documents_all_files / len(json_files):.1f}")
    print(f"Average chunks per document:       {total_chunks_all_files / total_documents_all_files:.1f}")
    print("=" * 80)
    
    return file_stats

def count_chunks_with_detailed_breakdown(folder_path):
    """
    More detailed analysis showing chunk count per document within each file.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return
    
    print("=" * 80)
    print(f"DETAILED CHUNK COUNT ANALYSIS - Folder: {folder_path}")
    print("=" * 80)
    print()
    
    total_chunks_all = 0
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                documents = data
            else:
                documents = [data]
            
            print(f"📄 File: {json_file.name}")
            print(f"   Total documents: {len(documents)}")
            
            file_chunks = []
            for idx, doc in enumerate(documents):
                chunks = doc.get('chunks', [])
                chunk_count = len(chunks)
                file_chunks.append(chunk_count)
                total_chunks_all += chunk_count
                
                # Get document title if available
                title = doc.get('original_document', {}).get('title', 'Untitled')
                if len(title) > 50:
                    title = title[:47] + "..."
                
                print(f"      Doc {idx+1}: {chunk_count:3d} chunks - {title}")
            
            print(f"   Total chunks in this file: {sum(file_chunks)}")
            print(f"   Average chunks per document: {sum(file_chunks)/len(documents):.1f}")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
    
    print("=" * 80)
    print(f"✅ GRAND TOTAL: {total_chunks_all} chunks across all files")
    print("=" * 80)

