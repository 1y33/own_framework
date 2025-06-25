import parse_files
import create_dataset
import tokenizer

directory =  "director_random"

texts = parse_files.parse_directory(directory)
word_count = parse_files.count_words(texts)
word_stats = parse_files.print_word_statistics(word_count)

toker = tokenizer.get_tokenizer()
ds = create_dataset.TextDataset(texts=texts,tokenizer=toker)

print(len(ds))
print(ds[100])
print(ds[100]["input_ids"].size())
print(toker.decode(ds[100]["input_ids"]))

