from pyspark.sql import SparkSession
import math

spark = SparkSession.builder.getOrCreate()

file = spark.read.text("project2_data.txt")

doc_count = file.count()

docs_terms = {}

# How many lines at most to be processed per iteration; to prevent OOM errors
lines_per_partition = 1500

print(doc_count, "document(s) to be processed with up to", lines_per_partition, "line(s) per partition")
print(math.ceil(doc_count / lines_per_partition), "partition(s) total")
print()

p_count = 0
while file.count() > 0:

    file_partition = file.limit(lines_per_partition)
    file_partition_p = file_partition.toPandas()

    # Term frequency (tf) for each term and its group of documents
    for row in file_partition_p.value:
        row_arr = row.strip().replace("  ", " ").split(" ")
        row_len = len(row_arr)
        row_docname = row_arr[0]
        row_tf_frac = 1 / (row_len - 1)
        for i in range(1, row_len):
            row_term = row_arr[i]
            if len(row_term) > 0:
                if docs_terms.get(row_term) == None:
                    docs_terms[row_term] = {}
                    docs_terms[row_term][row_docname] = row_tf_frac
                elif docs_terms[row_term].get(row_docname) == None:
                    docs_terms[row_term][row_docname] = row_tf_frac
                else:
                    docs_terms[row_term][row_docname] += row_tf_frac

    file = file.subtract(file_partition)

    p_count += 1
    print("Partition", p_count, "processed")

# tf * Inverse document frequency for each term and its group of documents
# doc_count is the size of the documents
# len(docs_terms[terms]) is the number of documents associated with each term
for terms in docs_terms:
    idf = math.log10(doc_count / len(docs_terms[terms]))
    for docnames in docs_terms[terms]:
        docs_terms[terms][docnames] *= idf

print()

while True:

    query = input("Enter a word: ")
    query_type = input("Enter 0 if querying for tf-idf; anything else if querying for term-term relevance: ")

    if docs_terms.get(query) == None:
        print("Not Found")
    else:
        if query_type == "0":
            for doc in docs_terms[query]:
                print(doc, docs_terms[query][doc])
        else:
            # Term-term frequency for each query / term pair, sorted in descending order
            query_len = len(query)

            term_term = []
            
            for terms in docs_terms:
                if terms != query:
                    terms_len = len(terms)
                    compare_size = query_len if query_len < terms_len else terms_len
                    same = 0
                    for i in range(compare_size):
                        if terms[i] == query[i]:
                            same += 1
                    term_term.append((terms, math.acos(same / (query_len * terms_len))))
            
            term_term.sort(key = lambda t : t[1], reverse = True)
            
            for terms in term_term:
                print(terms[0], terms[1])
