# replace .... with your user on windows if you want to run this file.
abs_path = 'C:\\Users\\....\\Desktop\\Fraud classification\\wiki_corpus_files'
file_paths = ["\\to0.txt", "\\to1.txt", "\\to2.txt", "\\to3.txt", "\\to4.txt", "\\to6.txt", "\\to7.txt", "\\to9.txt", 
         "\\to10.txt", "\\to11.txt", "\\to12.txt", "\\to13.txt", "\\to14.txt", "\\to15.txt"]

output_file = 'C:\\Users\\....\\Desktop\\Fraud classification\\wiki_corpus_files\\wikipedia_corpus.txt'



with open(output_file, 'w', encoding='utf-8') as out_file:
    for file_path in file_paths:
        print(f'processing file: {file_path}')
        with open(f'{abs_path}{file_path}', 'r', encoding='utf-8') as in_file:
            for line in in_file:
                out_file.write(line)
