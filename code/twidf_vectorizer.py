import itertools
import igraph
import copy
import numpy as np
import igraph


class TwIdfVectorizer():
    def __init__(self, w=5, b=0.003):
        self.w = w
        self.b = b
        
    def terms_to_graph(self, lists_of_terms, window_size, overspanning):
        if overspanning:
            terms = [item for sublist in lists_of_terms for item in sublist]
        else:
            idx = 0
            terms = lists_of_terms[idx]
        from_to = {}
        while True:
            w = min(window_size, len(terms))
            terms_temp = terms[0:w]
            indexes = list(itertools.combinations(range(w), r=2))
            new_edges = []
            for my_tuple in indexes:
                new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
            for new_edge in new_edges:
                if new_edge in from_to:
                    from_to[new_edge] += 1
                else:
                    from_to[new_edge] = 1

            for i in iter(range(w, len(terms))):
                considered_term = terms[i]
                terms_temp = terms[(i - w + 1):(i + 1)]
                candidate_edges = []
                for p in iter(range((w - 1))):
                    candidate_edges.append((terms_temp[p], considered_term))

                for try_edge in candidate_edges:
                    if try_edge[1] != try_edge[0]:
                        if try_edge in from_to:
                            from_to[try_edge] += 1
                        else:
                            from_to[try_edge] = 1
            if overspanning:
                break
            else:
                idx += 1
                if idx == len(lists_of_terms):
                    break
                terms = lists_of_terms[idx]
        g = igraph.Graph(directed=True)
        if overspanning:
            g.add_vertices(sorted(set(terms)))
        else:
            g.add_vertices(sorted(set([item for sublist in lists_of_terms for item in sublist])))

        g.add_edges(list(from_to.keys()))
        g.es['weight'] = list(from_to.values())  # based on co-occurence within sliding window
        g.vs['weight'] = g.strength(weights=list(from_to.values()))  # weighted degree

        return (g)

    def compute_node_centrality(self, graph):
        w_closeness = graph.closeness(normalized=True, weights=graph.es["weight"])
        w_closeness = [round(value,5) for value in w_closeness]
        return {k:v for k,v in zip(graph.vs["name"],w_closeness)}
        #return(zip(graph.vs["name"],w_closeness))
        
    def fit(self, docs):
        terms_by_doc = [document.split(" ") for document in docs]
        n_terms_per_doc = [len(terms) for terms in terms_by_doc]
        all_terms = [terms for sublist in terms_by_doc for terms in sublist]
        self.avg_len = sum(n_terms_per_doc)/len(n_terms_per_doc)
        self.vocab_ = list(set(all_terms))
        terms_by_doc_sets = [set(elt) for elt in terms_by_doc]
        n_doc = len(list(docs))
        self.idf = dict(zip(self.vocab_,[0]*len(self.vocab_)))

        for counter,unique_term in enumerate(self.idf.keys()):
            df = sum([unique_term in terms for terms in terms_by_doc_sets])
            self.idf[unique_term] = np.log10(float(n_doc+1)/df)    
        
    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)
    
    def transform(self, docs):
        terms_by_doc = [document.split(" ") for document in docs]
        all_graphs = []
        for elt in terms_by_doc:
            all_graphs.append(self.terms_to_graph([elt],self.w,overspanning=True))
        docs_transformed = []
        for i, graph in enumerate(all_graphs):
            terms_in_doc = [term for term in terms_by_doc[i] if term in self.vocab_]
            doc_len = len(terms_in_doc)
            my_metrics = self.compute_node_centrality(graph)
            feature_row_w_closeness = [0]*len(self.vocab_)
            for term in list(set(terms_in_doc)):
                index = self.vocab_.index(term)
                idf_term = self.idf[term]
                denominator = (1-self.b+(self.b*(float(doc_len)/self.avg_len)))
                #metrics_term = [e[1] for e in list(my_metrics) if e[0]==term][0]
                metrics_term = my_metrics[term]
                feature_row_w_closeness[index] = (metrics_term/denominator) * idf_term

            docs_transformed.append(feature_row_w_closeness)

        return np.nan_to_num(np.array(docs_transformed))
