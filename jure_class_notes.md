#### top

TOC:<br>
[Jure Lectures](#Jure-Leskovec-Stanford-CS224W-Fall-2019)<br>
- [Lecture 2 Properties of Networks](#Lecture-2-Properties-of-Networks-and-Random-Graph-Models:)<br>
- [Lecture 3 Structural Roles](##Lecture-3-Structural-Roles:)<br>
- [Lecture 4 Community Structure](#Lecture-4-Community-Structure:)<br>
- [Lecture 5 Spectral Clustering](#Lecture-5-Spectral-Clustering:)<br>
- [end of section](#endJureClass)

[statistics grouping](#Graph-Properties)




[back to top](#top)

## Classic Graph ML Tasks
    - Node Classification: Predict Property of Node
    - Link Prediction: Predict whether there are missing links betweeen two nodes
    - Graph classification: Categorize different graphs
    - Clustering: Detect if nodes form a community

    source:

## Additional Concepts

    Strongly connected directed graph:
    has a path from each node to every other node and vis versa

    Weakly connected directed graph:
    is connected if we disregard the edge directions

    Strongly connected components SCCs:
    subgraphs that are strongly connected, nodes can reach in in-component, out otherwise


# [Jure-Leskovec-Stanford-CS224W-Fall 2019](https://www.youtube.com/playlist?list=PL-Y8zK4dwCrQyASidb2mjj_itW2-YYx6-)

Notes TOC:<br>
[Lecture 2 Properties of Networks](#Lecture-2-Properties-of-Networks-and-Random-Graph-Models:)<br>
[Lecture 3 Structural Roles](##Lecture-3-Structural-Roles:)<br>
[Lecture 4 Community Structure](#Lecture-4-Community-Structure:)<br>
[Lecture 5 Spectral Clustering](#Lecture-5-Spectral-Clustering:)<br>
[end of section](#endJureClass)


## Lecture 2 Properties of Networks and Random Graph Models:
    https://www.youtube.com/watch?v=erMiEFGRsIk&list=PL-Y8zK4dwCrQyASidb2mjj_itW2-YYx6-&index=2

    Describe properties of entire graph and expected properties of a "random" graph for comparison

### 1. P(k) - Degree Distribution
<br>

    - P(k) = Nk / N
    - Nk = number of nodes with degree k
    - Directed, in-degree and out-degree distributions

    Probability that a random node has degree k.

    Directed graphs have in-degree distribution and out-degree distribution.

### 2. Paths
<br>

    How do we think of distance in a graph?
    - A collection of paths where a path is a collection of verticies

    - h Charactization of distribution of shortest path lengths
    - directed graphs, distance is not symmetric

    Shortest path:
        - unweighted: min(number of edges to traverse)
        - weighted: min(sum of weight edges to traverse)

    Diameter:
        - the maximum distance between any pair of nodes in a graph
        - fragile to a long tenticle off a well connected network

    Average path length:
        - h = ( 1 / (2 * max number of edges ) ) * sum shorted distance between all pairs of nodes
        - for a disconnected graph, you would have to skip those disconnected pairs or assume 0
        - assume connected graph or do computation over connected components of graph
    -

### 3. C - Clustering Coefficient
<br>

    - How connected are a node's neighbors?
    - C in the continuous set [0,1]
    - can have same degree with different clustering coefficient

    Ci = (2 * # edges between neighbors of i) / max # of edges between i's neighbors
    Ci = (2 * e) / ki * (ki - 1)
    where k is node i's degree

    - average clustering coefficient
        - simple average of C, 1/N sumN(C)

    - undefined for nodes with degree 0 or 1

### 4. Connectivity
<br>

    - Size of the largeset connected component
    - what fraction of nodes are in the largest connected component
    - s    - Distribution of connected component sizes
    - for directed graphs we have weak and strong connected components
        - not sure what the definitions are yet


### 5. Erdos-Renyi Random Graphs

<br>

    When we calculate statistics of a graph, we would like these numbers to a baseline.
    By building a graph with similar input statistics, degree distribution, node size, etc.
    We have something we can compare of graph of interest against.

    - Gnp, undirected n nodes w/ edge (u,v) iid with prob p
        1. degree distribution is binomial
            - mean = p(n-1)
            - variance = p(1-p)(n-1)

        - how does variance change as a fraction of mean?
            - by law of large numbers, as the graph size increases, the distribution becomes increasingly narrow(more leptokurtic)- we are increasingly confident that the degree of a node is in the vicinity of the mean
            - the fraction variance / mean will also go to zero
            - 1 / sqrt(n-1)

        - clustering coefficient
        - E[number of edges between neighbors] = p ((k * k-1)/2)
        - each pair is connected with prob p times the number of distinct pairs of neighbors of node i of degree k
        - E[C] = average degree / number of nodes = p
    - Gnm, undirected n nodes w/ m edges picked uniformly at random



    Interesting charts:

    - scatterplot of average clustering coefficient y axis for a degree on x axis

    - scatterplot of count of components y axis again component size x axis

    - scatterplot(distribution) of count of pairs y axis against shortest path lengths
        - avg path length
        - x% of all nodes can be reached in less than y hops
        - frontier of BFS two columns of values, steps and # of nodes
[back to top](#top)

## Lecture 3 Structural Roles:
    https://www.youtube.com/watch?v=sdpqpj8g6YY&list=PL-Y8zK4dwCrQyASidb2mjj_itW2-YYx6-&index=3

    Analysis of sub-network characterisitcs - specific approach at a mesocopic

    Describe network around the node, decompose into building blocks (graphlets)

    Mesoscopic view, but in the view of graphlets

    graphlets: non-isomorphic graphs of n nodes, 3=trianlges, 4=rectangles, etc<br>
    I think another definition could be induced egonet of n nodes


### 1. Subgraphs

    motif: recurring, significant patterns of interconnections, can overlap
    pattern: small induced subgraph
    recurring: high frequency
    significant: more frequent than expected(random network)
    induced: must include all edges on sub-graph

#### Motif Z Score:
    normalizing observed against random model
    Z = (Nreal - ExpectedNumRandom ) / std(ExpectedNumRandom)
    z score where we standardize actual count less expected count by std of expected count
    expected value is derived from a random network

    Network significance profile = feature vector with values for all subgraph types
    vector of normalized Z scores

#### Configuration Model:
    Generate a random graph with a given degree sequence (from real to calculate Z scores)
<br>

### 2. Structural roles in networks
<br>

    Graphlet: connected non-isomorphic subgraphs
    induced subgraphs of any frequency
    position in a given graphlet is a 'structural role'

    GDV graphlet degree vector: counts number of graphlets a node touches

    Orbits define the “roles” of the nodes within the graphlet.

    node A:
    orbit
    -a--b--c
    --1--2--1--0
    --2--0--1--3
    --3--2--2--0

    the GDV provides a measure of the node's local network topology

    role - group of nodes with similar structural properties
    communities - group of nodes that are well connected

    Roles and Communities are complementary

    structural equivalence: if two nodes have same relationship to all other nodes


### 3. Discovering structural roles and its applications
<br>

    Role query - similar behavoir to known target
    Role outliers - anamoly detection
    Role dynamics - unusual changes in behavior
    identity resolution - identify, de-anonymize, individuals in a new network
    role transfer - use knowledge of one network to make predictions of another
    network comparison - compute similarity of networks, determine compatibility for knowledge transfer

    Recursive Feature Extraction:

    turn network into a node x feature matrix, unsupervised classification on matrix

    local features: degree

    ego features: edges within egonet, in/out of egonet
    egonet is a induced subgraph of node and neighbors

    use aggregate features of a node to generate new recursive features
    such as sum and mean
    need to prune as recursive features grow - curse of dimmentionality

    can also use graphlet representations or a node x feature matrix on graphlets

[back to top](#top)
## Lecture 4 Community Structure:
<https://www.youtube.com/watch?v=Q7CHFo8UdPU&list=PL-Y8zK4dwCrQyASidb2mjj_itW2-YYx6-&index=4>



### Motivation to detect communities

     How does information flow through the network?
     What roles do different edges play? (short vs long)
     Two aspects to an edge (according to sociologist in 1960's):
        - structural - spans different structural parts of the network
            Triadic Closure: if two nodes have an edge with a node in common, increased
            likelihood the two nodes will have an edge as well
        - bilateral - strong or weak edge

    takeaways:
        1. structure
            structurally embedded edges are also socially strong
            long range edges spanning different parts of network are socially weak
        2. information
            stuctural embedded edges information are highly redundant
            long range edges allow nodes to access information in diff part of network

    definition: edge overlap == fraction of similar neighbors

### Modularity

    goal: define sets of node with many internal and few external connections

    Define: Modularity Q
        Measure of how well a network is partitioned into communities
        **idea** if there are more edges than expected we found a grouping deviating from randomness
        Q = # of edges in group - Expected number of edges in group (null model)
        S = partition: put every node into exactly one group
        where null model is a multiGragh w/ same degree distribution but uniformly random connections

        notation:
            E = summation (i lazy ok)
            Exp = would have to be expectation (because E() is usually expectation?? bleh)

        property: sum of node degree equals two times edges as every edge will be counted twice
            E(i in N) ki = 2m
            k = node degree
            N = set of nodes
            m = number of edges

        expected number of edges uniform between i, j
             = ki * kj / 2m
            we have k * k opportunities to make connections uniformly over all possible edges 2m


        Q(G,S) = 1/2m E(s in S) E(i in S) E(j in S) (Aij - kikj/2m)

        Q = modularity = takes values: [-1, 1], b/c we are scaling by 2m
        G = multigraph
        S = set of partitions
        s = single partition in set

        1/2m = normalize by total number of edges, because we are summing over two nodes degrees,
            we will visit each edge between the nodes twice, so we divide by 2 * m
        n = number nodes
        m = number of edges
        i, j = nodes in G
        E(s in S) E(i in S) E(j in S) = two nodes in same partition

        Aij = actual number of edges between nodes i and j
        ki*kj / 2m = expected number of nodes in between nodes i and j w/ uniform selection
            by changing the null model this term would change

        how to interpret Q?
            positive value more edges than expected = community
            negative value less edges than expected = anti-community
                cluster of nodes that should connect but don't

            Jure intuition: negative to positve value not congruent
                there don't have to be communities in graphs with negative Q if flipped





### Louvain Algorithm - 1st community detection algo

    How do we find communities?

    finds modularity in decreasing number of communities, pick max(Q) for iteration

    Louvain algo greedily maximizes modeularity
    n log n runtime
    most expensive will be first phase and then the aggregation decreases degree and speeds up

    pass through nodes in G
        start with every node in own communitiy
        phase 1: local changes, moving nodes to other communities if increases modularity
            until change in modularity doesn't increase
        phase 2: communitiy aggregation, nodes in each community are combined into a "super node"
        got to phase 1, ...

    ** output of algorithm depends on the order in which nodes are considered
        generally this doesn't matter when networks are large as shown in research




### detecting overlapping communities- BigCLAM

    overlapping communities is a bit trickier

    implementation requires number of communities to be picked at start

    high level :
        step 1: generate graphs
            define generative model for graphs AGM - community affiliation graph model

        step 2: maximum likelihood estimation
            given graph G, make assumption G was created by AGM
            find best AGM that could have generated G

    AGM - Community Affiliation Graph Model
        G = graph represented as bipartite
        C = communities nodes of bipartite representation
            p = likelihood of nodes in commnity to link to each other
        V = nodes
        M = membership edges between C and V

        given parameters (V, C, M, {pc})
            nodes in community c connect to each other with probability p
            basically an erdos-reynia model for each community
            nodes in multiple communties have multiple probabilites
            the more communities two nodes have in common the higher prob they are connected
            too lazy to do the math rn soz take word for it



    Maximum Likelihood Estimation:
    G -> bipartite G of nodes and communities

        How to estimate model F parameters
            given graph G, find model F
                affiliation graph M
                number of communities C
                parameters p

    arg max prob( G | Gf <- F)
        select maximum probability of graph G being similar to model F output Gf

    need to:
        efficiently calculate prob(G|F)
        search max over F (using gradient descent)

    think of G as adjacency matrix of 0,1's
    think of F producing matrix of probabilities for each edge for this adjacency matrix
        based on model F parameters, basically a sequence of beurnouli coin flips
        so prob(G|F) is the likelihood of getting G adj matrix from F prob adj matrix
        optimizing for prob matrix of F to get G adj matrix

    adj matrix G =
    [[0, 1],
     [1, 1]]

    model prob matrix =
    [[.05, .60],
     [.10, .25]]

    the prob of graph given model is
    product of probabilities that model generated each edge in graph
    product over all edge of graph (where == 1 in adj matrix G )
    product of probabilities that model generated each edge !in graph

    P(G|F) = product(u,v in G) {P(u,v)} * product(u,v !in G) {1 - P(u,v)}
        product(u,v in G) = nodes in graph G
        product(u,v !in G) = nodes not in graph G
        {P(u,v)} = probability of model F generating edge
        {1 - P(u,v)} = probability of model F not generating edge

    Relaxing AGM towards P(v,u)

        need to make bipartite representation more tractable, turn into nodes V into vectors
        Fu: a row vector of communitiy memberships of node u
            where component of vector measures strength of nodes connection to each community
            0 = not in community
            Fu = [FuA node u community A, FuB, ]
            Fv = [FvA, FvB, ] ... etc

        prob of nodes u, v linking is proportional to strengths of shared relationships
        P(u,v) = 1 - exponential of (-Fu dotproduct Transposed(Fv))

            dot product of two vectors is multiplied their coordinates and summing results
            [2, 3] dotprod [5, 4] = [10, 12] = 22

            if one of the entries is zero, the then dotprod will be zero
            coordinate will only be non-zero where both nodes belong to

            strength will be proportional to strength of memberships
            if they belong to more together, the prob will be higher

            if take negative and exponentiate it, we will get something very small
            1 - something small is a large probability

        but this is unstable as we are taking products of small probabilities
        so we take log likelihood

        l(F) = E(u,v in G)  log(1-exponential(-Fu dotproduct Transposed(Fv))
             - E(u,v !in G) Fu Transposed(Fv)
            1-exponential(-Fu dotproduct Transposed(Fv) = probability of the edge

            ran out of time and didn't really get all the explination for last equation

        Optimization:
            start with random F
            update FuC for node u while fixing the memberships of other nodes
            updating takes linear time in the degree of u


### additional thoughts on interpreting modularity

    on anamoly detection:
        if we have a group of nodes with a significat negative modulariy, < -0.3

        then this says that this group SHOULD have more connections due to a null model

        while this could be interpretted as anamalous, it isn't quite anamaloy detection

        more just statistical observation to look more closley at

        BUT tempting to wave around as anamoly detection

    on clustering:
        this is clustering, just with a specific objective, to maximize modularity

[back to top](#top)



## Lecture 5 Spectral Clustering:
<br>

https://www.youtube.com/watch?v=VIu-ORmRspA&list=PL-Y8zK4dwCrQyASidb2mjj_itW2-YYx6-&index=5


    blah blah blah






[back to top](#top)

### endJureClass

## Graph Properties
    <-- not Jure Class

    To organize:
        Characteristic path length

    1. Microscopic - local features
        degreee distribution
        local clustering, egonet
        connectivity
            nx.algorithms.approximation.all_pairs_node_connectivity()
            gives an strict lower bound on the actual number of node independent paths between two nodes
            nx.algorithms.approximation.connectivity.node_connectivity()
            macroscopic
        eccentricity
        periphery
        pageRank


    2. Mesoscopic - community level
        modularity
        graphlet

    3. Macroscopic
        betweeness centrality
        eigenvector centrality
        mean shortest path length
        assortativity coefficient
            connectivity tendency
            https://arxiv.org/abs/1212.6456
        connectivity
            nx.algorithms.approximation.connectivity.node_connectivity()


    1. Clustering
        1. Clustering Coefficient
        2. centrality measure

    2. Paths
        1. shortest paths
        2. uniformity of shortest path lengths is a graph uniformity measure
        3. path prediction with lags? lagged path correlations

    3. Triads
        1. Open, indicates holes in flow
        2. Closed, indicates mutual flow

    4. Modularity
        1. measure of how compartmentalized a graph is
        2. ClausetNewmanMoore algorithm
        3. score close to 1 is high modularity implying each nodes belongs to a specific cluster

    5. Evolution
        1. Observe basic properties over time
        2. Multiplex


 reference: https://networkx.github.io/documentation/stable/_downloads/networkx_reference.pdf



Message Passing Models:

https://arxiv.org/pdf/2009.03509.pdf

Two Kinds:

1. GNN
    combine graph structures by propagating and aggregating node features through several neural layers, which get predictions from feature propagation

2. Label Propagation Algorithm
    makes predictions for unlabeled instances(nodes?) by iterative label propagation

Based on same assumption of making semi-supervised classifications by information propagation

problem, different vector spaces
node feature: embedding
node label: one-hot vector

Unified Message Passing Model UniMP: multi-layer graph transformer

jointly using:
1. label embedding to transform node labels into same vector space as node features
2. propagates node features, multi-head attentions are used as transition matrix for propagating label vectors


