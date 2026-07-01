# Part C - Reflection

If users report outdated or irrelevant answers even though the source documents are correct, I would first check whether the system is actually searching the latest index. I would compare the active index version against the latest approved document manifest, then replay affected queries and inspect the retrieved top-k chunks, source document IDs, timestamps, and similarity scores. This separates a stale-index problem from a retrieval-quality problem.

Second, I would inspect chunking and metadata. The source document can be correct but still not retrievable if the relevant section was split badly, embedded without enough surrounding context, or missing metadata such as department, policy version, or effective date. I would trace failed queries back to the expected source passages and check whether those passages exist as searchable chunks with the right metadata.

Third, I would look at query ambiguity and ranking. Users may ask broad questions like "What is the approval process?" when several departments have similarly named policies. I would review query logs, citation clicks, thumbs down feedback, and reformulated queries, then test whether department filters, hybrid keyword-plus-vector retrieval, or a lightweight reranker improves top-k recall on a small labeled query set.
