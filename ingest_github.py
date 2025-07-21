from gitingest import ingest
import json
import re
from tqdm import tqdm
import os

output_dir = 'dataset/ingested_content_chunks'
os.makedirs(output_dir, exist_ok=True)

your_json = " "
with open('your_json') as f:
    repos = json.load(f)

all_repos = repos['hip_repos'] + repos['cuda_repos']

total_tokens = 0
for i, repo_url in enumerate(tqdm(all_repos, desc="Processing repositories")):
    try:
        summary, tree, content = ingest(repo_url,
                                        exclude_patterns="*.pdf,*.txt,*.pptx,*.cfg,*.json,*.yml,*.xml",
                                        include_patterns="*.py,*.cpp,*.c,*.h,*.hpp,*.md,*.hip,*.cu")

        match = re.search(r'Estimated tokens:\s*([\d\.]+)\s*([kM])', summary, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit == 'k':
                tokens = value * 1_000
            elif unit == 'm':
                tokens = value * 1_000_000
            else:
                tokens = value
            total_tokens += tokens

        chunk_size = 16 * 1024  # 16 KB
        repo_name_safe = re.sub(r'[^\w\-_\.]', '_', repo_url) # Sanitize repo name for filename
        
        chunk_base_filename = os.path.join(output_dir, f"repo_{i}_{repo_name_safe}")

        for j in range(0, len(content), chunk_size):
            chunk = content[j:j + chunk_size]
            chunk_filename = f"{chunk_base_filename}_chunk_{j // chunk_size}.txt"
            with open(chunk_filename, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)

    except Exception as e:
        print(f"Error processing {repo_url}: {e}")

print(f"Total estimated tokens: {int(total_tokens):,}")