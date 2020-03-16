# -*- coding: utf-8 -*-

# @Time    : 2020/2/3
# @Author  : yeyuc
# @Email   : yeyucheng_uestc@163.com
# @File    : yeyuc_networkx.py
# @Software: PyCharm

"""
概要：整理networkx的常用库函数，实现复杂网络中的一些指标，并封装成类NetworkxPython

使用方法：

    步骤1：继承NetworkxPython，实例化Successor，示例：client = Successor()
    步骤2：重载network，生成网络，返回图对象G
    步骤3：重载job，使用图对象，计算所需指标，完成任务

常用函数：

    # 图对象 ---------------------------------------------------------------------------------------------------
    # client.rg(d=3, n=20)  # 规则网络
    # client.er(n=20, p=0.2)  # 随机网络
    # client.ws(n=20, k=4, p=0.3)  # 小世界网络
    # client.ba(n=20, m=2)  # 无标度网络
    # client.digraph(data, is_weighted=False)  # 导入数据，生成有向（含权）网络
    # client.graph(data, is_weighted=False)  # 导入数据，生成无向（含权）网络

    # 常用图属性 -------------------------------------------------------------------------------------------------
    # client.is_directed()  # 判断是否为有向图
    # client.neighbor()  # 返回节点邻居
    # client.in_neighbor()  # 返回节点入邻居
    # client.out_neighbor()  # 返回节点出邻居
    # client.connected_components()  # 默认返回所有连通片，max=True，则返回最大连通片

    # 常用库指标 -------------------------------------------------------------------------------------------------
    # client.degree()  # 返回节点度
    # client.in_degree()  # 返回节点入度
    # client.out_degree()  # 返回节点出度
    # client.strength()  # 返回节点强度
    # client.in_strength()  # 返回节点入强度
    # client.out_strength()  # 返回节点出强度
    # client.pagerank(alpha=0.85)  # 返回节点的PR值
    # client.clustering()  # 返回节点的聚类系数
    # client.average_clustering()  # 计算网络的平均聚类系数
    # client.degree_centrality()  # 返回节点度中心性
    # client.in_degree_centrality()  # 返回节点入度中心性
    # client.out_degree_centrality()  # 返回节点出度中心性
    # client.closeness_centrality()  # 返回节点接近中心性
    # client.betweenness_centrality()  # 返回节点介数中心性
    # client.eigenvector_centrality()  # 返回节点特征向量中心性
    # client.degree_assortativity_coefficient()  # 计算网络的同配系数

    # 常用科研指标 -----------------------------------------------------------------------------------------------
    # client.h_index()  # 返回节点h-index值
    # client.CI()  # 返回节点CI值（网络传播，collective influence）
    # client.CI_in()  # 返回节点入向CI值
    # client.CI_out()  # 返回节点出向CI值
    # client.core()  # 返回节点核数值
    # client.in_core()  # 返回节点的入向核数值
    # client.out_core()  # 返回节点的出向核数值
    # client.core(weighted=True)  # 返回节点含权核数值
    # client.in_core(weighted=True)  # 返回节点的入向含权核数值
    # client.out_core(weighted=True)  # 返回节点的出含权向核数值

注：上述函数，:return None/dict; float(平均聚类系数, 同配系数)
"""

import random
import networkx as nx
from lib.yeyuc_logging import LoggingPython
from lib.common import Common


# 建立图，返回图对象
class Network():

    # 规则网络，随机网络，小世界网络，无标度网络 -----------------------------------------------------------------
    def rg(self, d=3, n=20):
        r"""Returns a random $d$-regular graph on $n$ nodes.

        The resulting graph has no self-loops or parallel edges.

        Parameters
        ----------
        d : int
          The degree of each node.
        n : integer
          The number of nodes. The value of $n \times d$ must be even.
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Notes
        -----
        The nodes are numbered from $0$ to $n - 1$.

        Kim and Vu's paper [2]_ shows that this algorithm samples in an
        asymptotically uniform way from the space of random graphs when
        $d = O(n^{1 / 3 - \epsilon})$.

        Raises
        ------

        NetworkXError
            If $n \times d$ is odd or $d$ is greater than or equal to $n$.

        References
        ----------
        .. [1] A. Steger and N. Wormald,
           Generating random regular graphs quickly,
           Probability and Computing 8 (1999), 377-396, 1999.
           http://citeseer.ist.psu.edu/steger99generating.html

        .. [2] Jeong Han Kim and Van H. Vu,
           Generating random regular graphs,
           Proceedings of the thirty-fifth ACM symposium on Theory of computing,
           San Diego, CA, USA, pp 213--222, 2003.
           http://portal.acm.org/citation.cfm?id=780542.780576
        """

        RG = nx.random_graphs.random_regular_graph(d, n)
        return RG

    def er(self, n=20, p=0.2, **kwargs):
        """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
        or a binomial graph.

        The $G_{n,p}$ model chooses each of the possible edges with probability $p$.

        The functions :func:`binomial_graph` and :func:`erdos_renyi_graph` are
        aliases of this function.

        Parameters
        ----------
        n : int
            The number of nodes.
        p : float
            Probability for edge creation.
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
        directed : bool, optional (default=False)
            If True, this function returns a directed graph.

        See Also
        --------
        fast_gnp_random_graph

        Notes
        -----
        This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for
        small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.

        References
        ----------
        .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
        """

        ER = nx.random_graphs.erdos_renyi_graph(n, p, seed=kwargs.get('seed'), directed=kwargs.get('directed', False))
        return ER

    def ws(self, n=20, k=4, p=0.3):
        """Returns a Watts–Strogatz small-world graph.

        Parameters
        ----------
        n : int
            The number of nodes
        k : int
            Each node is joined with its `k` nearest neighbors in a ring
            topology.
        p : float
            The probability of rewiring each edge
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        See Also
        --------
        newman_watts_strogatz_graph()
        connected_watts_strogatz_graph()

        Notes
        -----
        First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
        to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
        Then shortcuts are created by replacing some edges as follows: for each
        edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
        with probability $p$ replace it with a new edge $(u, w)$ with uniformly
        random choice of existing node $w$.

        In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
        does not increase the number of edges. The rewired graph is not guaranteed
        to be connected as in :func:`connected_watts_strogatz_graph`.

        References
        ----------
        .. [1] Duncan J. Watts and Steven H. Strogatz,
           Collective dynamics of small-world networks,
           Nature, 393, pp. 440--442, 1998.
        """

        # 基于WS小世界模型
        SM = nx.random_graphs.watts_strogatz_graph(n, k, p)
        return SM

    def ba(self, n=20, m=2):
        """Returns a random graph according to the Barabási–Albert preferential
        attachment model.

        A graph of $n$ nodes is grown by attaching new nodes each with $m$
        edges that are preferentially attached to existing nodes with high degree.

        Parameters
        ----------
        n : int
            Number of nodes
        m : int
            Number of edges to attach from a new node to existing nodes
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        G : Graph

        Raises
        ------
        NetworkXError
            If `m` does not satisfy ``1 <= m < n``.

        References
        ----------
        .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
           random networks", Science 286, pp 509-512, 1999.
        """

        BA = nx.random_graphs.barabasi_albert_graph(n, m)

        return BA

    # 有向网络与无向网络 -----------------------------------------------------------------------------------------
    def digraph(self, data, is_weighted=False):
        """
        有向图
        :param is_weighted: 边是否含权
        :return: graph object G and nodes
        """
        G = nx.DiGraph()
        if is_weighted:
            for record in data:
                G.add_edge(record[0], record[1], weight=float(record[2]))
        else:
            for record in data:
                G.add_edge(record[0], record[1])
        return G

    def graph(self, data, is_weighted=False):
        """
        无向图
        :param is_weighted: 边是否含权
        :return: graph object G and nodes
        """
        G = nx.Graph()
        if is_weighted:
            for record in data:
                G.add_edge(record[0], record[1], weight=float(record[2]), capacity=15, length=342.7)
        else:
            for record in data:
                G.add_edge(record[0], record[1])
        return G

    # 网络属性 ---------------------------------------------------------------------------------------------------
    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return self.G.is_directed()

    def neighbor(self):
        """
        计算无向网络节点的邻居
        :return: dict
        """
        if self.is_directed():
            self.logger.warning('G为有向网络！请使用in_neighbor或者out_neighbor')
            return {}
        else:
            neighbors = {node: dict(self.G[node]) for node in self.G}
            return neighbors

    def in_neighbor(self):
        """
        计算有向网络节点的入邻居
        :return: dict
        """
        if self.is_directed():
            neighbors = {node: {n: self.G[n][node] for n in self.G if node in self.G[n]} for node in self.G}
            return neighbors
        else:
            self.logger.warning('G为无向网络！请使用neighbor')
            return {}

    def out_neighbor(self):
        if self.is_directed():
            neighbors = {node: dict(self.G[node]) for node in self.G}
            return neighbors
        else:
            self.logger.warning('G为无向网络！请使用neighbor')
            return False

    def connected_components(self, **kwargs):
        """Generate connected components.

        Parameters
        ----------
        G : NetworkX graph
           An undirected graph

        Returns
        -------
        comp : generator of sets
           A generator of sets of nodes, one for each component of G.

        Raises
        ------
        NetworkXNotImplemented:
            If G is directed.

        Examples
        --------
        Generate a sorted list of connected components, largest first.

        >>> G = nx.path_graph(4)
        >>> nx.add_path(G, [10, 11, 12])
        >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        [4, 3]

        If you only want the largest connected component, it's more
        efficient to use max instead of sort.

        >>> largest_cc = max(nx.connected_components(G), key=len)

        To create the induced subgraph of each component use:
        >>> S = [G.subgraph(c).copy() for c in connected_components(G)]

        See Also
        --------
        strongly_connected_components
        weakly_connected_components

        Notes
        -----
        For undirected graphs only.

        """

        if self.is_directed():
            self.G = self.G.to_undirected()
        if kwargs.get('max'):
            return max(nx.connected_components(self.G), key=len)  # 高效找出最大的联通成分，其实就是sorted里面的No.1
        else:
            return nx.connected_components(self.G)


# 库指标
class LibMetrics():

    # 度 ---------------------------------------------------------------------------------------------------------
    def degree(self):
        """
        :return: 无向网络度，dict
        """
        try:
            if self.G.is_directed():
                self.logger.warning('G为有向网络！')
            else:
                self.logger.info('正在计算无向网络的度 ...')
                return self.order_dict({record[0]: record[1] for record in self.G.degree}, index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def in_degree(self):
        """
        :return: 有向网络度入度，dict
        """
        try:
            if not self.G.is_directed():
                self.logger.warning('G为无向网络！')
            else:
                self.logger.info('正在计算有向网络的入度 ...')
                return self.order_dict({record[0]: record[1] for record in self.G.in_degree}, index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def out_degree(self):
        """
        :return: 有向网络度出度，dict
        """
        try:
            if not self.G.is_directed():
                self.logger.warning('G为无向网络！')
            else:
                self.logger.info('正在计算有向网络的出度 ...')
                return self.order_dict({record[0]: record[1] for record in self.G.out_degree}, index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    # 强度(含权度) -----------------------------------------------------------------------------------------------
    def strength(self, **kwargs):
        """
        :return: 无向网络强度，dict
        """
        try:
            if self.G.is_directed():
                self.logger.warning('G为有向网络！')
            else:
                if not kwargs.get("hide_log"):
                    self.logger.info('正在计算无向网络的强度 ...')
                strength = {record[0]: record[1] for record in self.G.degree(weight='weight')}
                if not kwargs.get("hide_log"):
                    self.logger.info('强度计算完成！ ...')
                return self.order_dict(strength, index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def in_strength(self, **kwargs):
        """
        :return: 有向网络度入强度，dict
        """
        try:
            if not self.G.is_directed():
                self.logger.warning('G为无向网络！')
            else:
                if not kwargs.get("hide_log"):
                    self.logger.info('正在计算有向网络的入强度 ...')
                instrength = {record[0]: record[1] for record in self.G.in_degree(weight='weight')}
                if not kwargs.get("hide_log"):
                    self.logger.info('入强度计算完成！ ...')
                return self.order_dict(instrength, index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def out_strength(self, **kwargs):
        """
        :return: 有向网络度出强度，dict
        """
        try:
            if not self.G.is_directed():
                self.logger.warning('G为无向网络！')
            else:
                if not kwargs.get("hide_log"):
                    self.logger.info('正在计算有向网络的出强度 ...')
                outstrength = {record[0]: record[1] for record in self.G.out_degree(weight='weight')}
                if not kwargs.get("hide_log"):
                    self.logger.info('出强度计算完成！ ...')
                return self.order_dict(outstrength, index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    # Link Analysis ----------------------------------------------------------------------------------------------
    def pagerank(self, alpha=0.85):
        """Returns the PageRank of the nodes in the graph.

            PageRank computes a ranking of the nodes in the graph G based on
            the structure of the incoming links. It was originally designed as
            an algorithm to rank web pages.

            Parameters
            ----------
            G : graph
              A NetworkX graph.  Undirected graphs will be converted to a directed
              graph with two directed edges for each undirected edge.

            alpha : float, optional
              Damping parameter for PageRank, default=0.85.

            personalization: dict, optional
              The "personalization vector" consisting of a dictionary with a
              key some subset of graph nodes and personalization value each of those.
              At least one personalization value must be non-zero.
              If not specfiied, a nodes personalization value will be zero.
              By default, a uniform distribution is used.

            max_iter : integer, optional
              Maximum number of iterations in power method eigenvalue solver.

            tol : float, optional
              Error tolerance used to check convergence in power method solver.

            nstart : dictionary, optional
              Starting value of PageRank iteration for each node.

            weight : key, optional
              Edge data key to use as weight.  If None weights are set to 1.

            dangling: dict, optional
              The outedges to be assigned to any "dangling" nodes, i.e., nodes without
              any outedges. The dict key is the node the outedge points to and the dict
              value is the weight of that outedge. By default, dangling nodes are given
              outedges according to the personalization vector (uniform if not
              specified). This must be selected to result in an irreducible transition
              matrix (see notes under google_matrix). It may be common to have the
              dangling dict to be the same as the personalization dict.

            Returns
            -------
            pagerank : dictionary
               Dictionary of nodes with PageRank as value

            Examples
            --------
            >>> G = nx.DiGraph(nx.path_graph(4))
            >>> pr = nx.pagerank(G, alpha=0.9)

            Notes
            -----
            The eigenvector calculation is done by the power iteration method
            and has no guarantee of convergence.  The iteration will stop after
            an error tolerance of ``len(G) * tol`` has been reached. If the
            number of iterations exceed `max_iter`, a
            :exc:`networkx.exception.PowerIterationFailedConvergence` exception
            is raised.

            The PageRank algorithm was designed for directed graphs but this
            algorithm does not check if the input graph is directed and will
            execute on undirected graphs by converting each edge in the
            directed graph to two edges.

            See Also
            --------
            pagerank_numpy, pagerank_scipy, google_matrix

            Raises
            ------
            PowerIterationFailedConvergence
                If the algorithm fails to converge to the specified tolerance
                within the specified number of iterations of the power iteration
                method.

            References
            ----------
            .. [1] A. Langville and C. Meyer,
               "A survey of eigenvector methods of web information retrieval."
               http://citeseer.ist.psu.edu/713792.html
            .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
               The PageRank citation ranking: Bringing order to the Web. 1999
               http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf

            """
        try:
            self.logger.info('正在计算网络的PageRank值 ...')
            return self.order_dict(nx.pagerank(self.G, alpha=alpha), index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    # Cluster ----------------------------------------------------------------------------------------------------
    def clustering(self, **kwargs):
        r"""Compute the clustering coefficient for nodes.

    For unweighted graphs, the clustering of a node :math:`u`
    is the fraction of possible triangles through that node that exist,

    .. math::

      c_u = \frac{2 T(u)}{deg(u)(deg(u)-1)},

    where :math:`T(u)` is the number of triangles through node :math:`u` and
    :math:`deg(u)` is the degree of :math:`u`.

    For weighted graphs, there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2]_,

    .. math::

       c_u = \frac{1}{deg(u)(deg(u)-1))}
             \sum_{vw} (\hat{w}_{uv} \hat{w}_{uw} \hat{w}_{vw})^{1/3}.

    The edge weights :math:`\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\hat{w}_{uv} = w_{uv}/\max(w)`.

    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

    For directed graphs, the clustering is similarly defined as the fraction
    of all possible directed triangles or geometric average of the subgraph
    edge weights for unweighted and weighted directed graph respectively [3]_.

    .. math::

       c_u = \frac{1}{deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\leftrightarrow}(u)}
             T(u),

    where :math:`T(u)` is the number of directed triangles through node
    :math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
    :math:`u` and :math:`deg^{\leftrightarrow}(u)` is the reciprocal degree of
    :math:`u`.

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute clustering for nodes in this container.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    out : float, or dictionary
       Clustering coefficient at specified nodes

    Examples
    --------
    >>> G=nx.complete_graph(5)
    >>> print(nx.clustering(G,0))
    1.0
    >>> print(nx.clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    Self loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Intensity and coherence of motifs in weighted complex
       networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
       Physical Review E, 71(6), 065103 (2005).
    .. [3] Clustering in complex directed networks by G. Fagiolo,
       Physical Review E, 76(2), 026107 (2007).
    """

        try:
            self.logger.info('正在计算网络的聚类系数 ...')
            return self.order_dict(nx.clustering(self.G, nodes=kwargs.get('nodes'), weight=kwargs.get('weight')),
                                   index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def average_clustering(self, **kwargs):
        r"""Compute the average clustering coefficient for the graph G.

            The clustering coefficient for the graph is the average,

            .. math::

               C = \frac{1}{n}\sum_{v \in G} c_v,

            where :math:`n` is the number of nodes in `G`.

            Parameters
            ----------
            G : graph

            nodes : container of nodes, optional (default=all nodes in G)
               Compute average clustering for nodes in this container.

            weight : string or None, optional (default=None)
               The edge attribute that holds the numerical value used as a weight.
               If None, then each edge has weight 1.

            count_zeros : bool
               If False include only the nodes with nonzero clustering in the average.

            Returns
            -------
            avg : float
               Average clustering

            Examples
            --------
            >>> G=nx.complete_graph(5)
            >>> print(nx.average_clustering(G))
            1.0

            Notes
            -----
            This is a space saving routine; it might be faster
            to use the clustering function to get a list and then take the average.

            Self loops are ignored.

            References
            ----------
            .. [1] Generalizations of the clustering coefficient to weighted
               complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
               K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
               http://jponnela.com/web_documents/a9.pdf
            .. [2] Marcus Kaiser,  Mean clustering coefficients: the role of isolated
               nodes and leafs on clustering measures for small-world networks.
               https://arxiv.org/abs/0802.2512
            """

        try:
            self.logger.info('正在计算网络的平均聚类系数 ...')
            return nx.average_clustering(self.G, nodes=kwargs.get('nodes'), weight=kwargs.get('weight'),
                                         count_zeros=kwargs.get('count_zeros', True))
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    # Centrality -------------------------------------------------------------------------------------------------
    def degree_centrality(self):
        """Compute the degree centrality for nodes.

        The degree centrality for a node v is the fraction of nodes it
        is connected to.

        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        nodes : dictionary
           Dictionary of nodes with degree centrality as the value.

        See Also
        --------
        betweenness_centrality, load_centrality, eigenvector_centrality

        Notes
        -----
        The degree centrality values are normalized by dividing by the maximum
        possible degree in a simple graph n-1 where n is the number of nodes in G.

        For multigraphs or graphs with self loops the maximum degree might
        be higher than n-1 and values of degree centrality greater than 1
        are possible.
        """

        try:
            if self.G.is_directed():
                self.logger.warning('G为有向网络！')
            else:
                self.logger.info('正在计算无向网络的度中心性 ...')
                return self.order_dict(nx.degree_centrality(self.G), index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def in_degree_centrality(self):
        """Compute the in-degree centrality for nodes.

        The in-degree centrality for a node v is the fraction of nodes its
        incoming edges are connected to.

        Parameters
        ----------
        G : graph
            A NetworkX graph

        Returns
        -------
        nodes : dictionary
            Dictionary of nodes with in-degree centrality as values.

        Raises
        ------
        NetworkXNotImplemented:
            If G is undirected.

        See Also
        --------
        degree_centrality, out_degree_centrality

        Notes
        -----
        The degree centrality values are normalized by dividing by the maximum
        possible degree in a simple graph n-1 where n is the number of nodes in G.

        For multigraphs or graphs with self loops the maximum degree might
        be higher than n-1 and values of degree centrality greater than 1
        are possible.
        """
        try:
            if not self.G.is_directed():
                self.logger.warning('G为无向网络！')
            else:
                self.logger.info('正在计算有向网络的入度中心性 ...')
                return self.order_dict(nx.in_degree_centrality(self.G), index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def out_degree_centrality(self):
        """Compute the out-degree centrality for nodes.

            The out-degree centrality for a node v is the fraction of nodes its
            outgoing edges are connected to.

            Parameters
            ----------
            G : graph
                A NetworkX graph

            Returns
            -------
            nodes : dictionary
                Dictionary of nodes with out-degree centrality as values.

            Raises
            ------
            NetworkXNotImplemented:
                If G is undirected.

            See Also
            --------
            degree_centrality, in_degree_centrality

            Notes
            -----
            The degree centrality values are normalized by dividing by the maximum
            possible degree in a simple graph n-1 where n is the number of nodes in G.

            For multigraphs or graphs with self loops the maximum degree might
            be higher than n-1 and values of degree centrality greater than 1
            are possible.
            """

        try:
            if not self.G.is_directed():
                self.logger.warning('G为无向网络！')
            else:
                self.logger.info('正在计算有向网络的出度中心性 ...')
                return self.order_dict(nx.out_degree_centrality(self.G), index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def closeness_centrality(self):
        r"""Compute closeness centrality for nodes.

            Closeness centrality [1]_ of a node `u` is the reciprocal of the
            average shortest path distance to `u` over all `n-1` reachable nodes.

            .. math::

                C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

            where `d(v, u)` is the shortest-path distance between `v` and `u`,
            and `n` is the number of nodes that can reach `u`. Notice that the
            closeness distance function computes the incoming distance to `u`
            for directed graphs. To use outward distance, act on `G.reverse()`.

            Notice that higher values of closeness indicate higher centrality.

            Wasserman and Faust propose an improved formula for graphs with
            more than one connected component. The result is "a ratio of the
            fraction of actors in the group who are reachable, to the average
            distance" from the reachable actors [2]_. You might think this
            scale factor is inverted but it is not. As is, nodes from small
            components receive a smaller closeness value. Letting `N` denote
            the number of nodes in the graph,

            .. math::

                C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

            Parameters
            ----------
            G : graph
              A NetworkX graph

            u : node, optional
              Return only the value for node u

            distance : edge attribute key, optional (default=None)
              Use the specified edge attribute as the edge distance in shortest
              path calculations

            wf_improved : bool, optional (default=True)
              If True, scale by the fraction of nodes reachable. This gives the
              Wasserman and Faust improved formula. For single component graphs
              it is the same as the original formula.

            Returns
            -------
            nodes : dictionary
              Dictionary of nodes with closeness centrality as the value.

            See Also
            --------
            betweenness_centrality, load_centrality, eigenvector_centrality,
            degree_centrality, incremental_closeness_centrality

            Notes
            -----
            The closeness centrality is normalized to `(n-1)/(|G|-1)` where
            `n` is the number of nodes in the connected part of graph
            containing the node.  If the graph is not completely connected,
            this algorithm computes the closeness centrality for each
            connected part separately scaled by that parts size.

            If the 'distance' keyword is set to an edge attribute key then the
            shortest-path length will be computed using Dijkstra's algorithm with
            that edge attribute as the edge weight.

            The closeness centrality uses *inward* distance to a node, not outward.
            If you want to use outword distances apply the function to `G.reverse()`

            In NetworkX 2.2 and earlier a bug caused Dijkstra's algorithm to use the
            outward distance rather than the inward distance. If you use a 'distance'
            keyword and a DiGraph, your results will change between v2.2 and v2.3.

            References
            ----------
            .. [1] Linton C. Freeman: Centrality in networks: I.
               Conceptual clarification. Social Networks 1:215-239, 1979.
               http://leonidzhukov.ru/hse/2013/socialnetworks/papers/freeman79-centrality.pdf
            .. [2] pg. 201 of Wasserman, S. and Faust, K.,
               Social Network Analysis: Methods and Applications, 1994,
               Cambridge University Press.
            """
        try:
            self.logger.info('正在计算网络的接近中心性 ...')
            return self.order_dict(nx.closeness_centrality(self.G), index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def betweenness_centrality(self):
        r"""Compute the shortest-path betweenness centrality for nodes.

            Betweenness centrality of a node $v$ is the sum of the
            fraction of all-pairs shortest paths that pass through $v$

            .. math::

               c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

            where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
            shortest $(s, t)$-paths,  and $\sigma(s, t|v)$ is the number of
            those paths  passing through some  node $v$ other than $s, t$.
            If $s = t$, $\sigma(s, t) = 1$, and if $v \in {s, t}$,
            $\sigma(s, t|v) = 0$ [2]_.

            Parameters
            ----------
            G : graph
              A NetworkX graph.

            k : int, optional (default=None)
              If k is not None use k node samples to estimate betweenness.
              The value of k <= n where n is the number of nodes in the graph.
              Higher values give better approximation.

            normalized : bool, optional
              If True the betweenness values are normalized by `2/((n-1)(n-2))`
              for graphs, and `1/((n-1)(n-2))` for directed graphs where `n`
              is the number of nodes in G.

            weight : None or string, optional (default=None)
              If None, all edge weights are considered equal.
              Otherwise holds the name of the edge attribute used as weight.

            endpoints : bool, optional
              If True include the endpoints in the shortest path counts.

            seed : integer, random_state, or None (default)
                Indicator of random number generation state.
                See :ref:`Randomness<randomness>`.
                Note that this is only used if k is not None.

            Returns
            -------
            nodes : dictionary
               Dictionary of nodes with betweenness centrality as the value.

            See Also
            --------
            edge_betweenness_centrality
            load_centrality

            Notes
            -----
            The algorithm is from Ulrik Brandes [1]_.
            See [4]_ for the original first published version and [2]_ for details on
            algorithms for variations and related metrics.

            For approximate betweenness calculations set k=#samples to use
            k nodes ("pivots") to estimate the betweenness values. For an estimate
            of the number of pivots needed see [3]_.

            For weighted graphs the edge weights must be greater than zero.
            Zero edge weights can produce an infinite number of equal length
            paths between pairs of nodes.

            The total number of paths between source and target is counted
            differently for directed and undirected graphs. Directed paths
            are easy to count. Undirected paths are tricky: should a path
            from "u" to "v" count as 1 undirected path or as 2 directed paths?

            For betweenness_centrality we report the number of undirected
            paths when G is undirected.

            For betweenness_centrality_subset the reporting is different.
            If the source and target subsets are the same, then we want
            to count undirected paths. But if the source and target subsets
            differ -- for example, if sources is {0} and targets is {1},
            then we are only counting the paths in one direction. They are
            undirected paths but we are counting them in a directed way.
            To count them as undirected paths, each should count as half a path.

            References
            ----------
            .. [1] Ulrik Brandes:
               A Faster Algorithm for Betweenness Centrality.
               Journal of Mathematical Sociology 25(2):163-177, 2001.
               http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
            .. [2] Ulrik Brandes:
               On Variants of Shortest-Path Betweenness
               Centrality and their Generic Computation.
               Social Networks 30(2):136-145, 2008.
               http://www.inf.uni-konstanz.de/algo/publications/b-vspbc-08.pdf
            .. [3] Ulrik Brandes and Christian Pich:
               Centrality Estimation in Large Networks.
               International Journal of Bifurcation and Chaos 17(7):2303-2318, 2007.
               http://www.inf.uni-konstanz.de/algo/publications/bp-celn-06.pdf
            .. [4] Linton C. Freeman:
               A set of measures of centrality based on betweenness.
               Sociometry 40: 35–41, 1977
               http://moreno.ss.uci.edu/23.pdf
            """

        try:
            self.logger.info('正在计算网络的介数中心性 ...')
            return self.order_dict(nx.betweenness_centrality(self.G), index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    def eigenvector_centrality(self):
        r"""Compute the eigenvector centrality for the graph `G`.

           Eigenvector centrality computes the centrality for a node based on the
           centrality of its neighbors. The eigenvector centrality for node $i$ is
           the $i$-th element of the vector $x$ defined by the equation

           .. math::

               Ax = \lambda x

           where $A$ is the adjacency matrix of the graph `G` with eigenvalue
           $\lambda$. By virtue of the Perron–Frobenius theorem, there is a unique
           solution $x$, all of whose entries are positive, if $\lambda$ is the
           largest eigenvalue of the adjacency matrix $A$ ([2]_).

           Parameters
           ----------
           G : graph
             A networkx graph

           max_iter : integer, optional (default=100)
             Maximum number of iterations in power method.

           tol : float, optional (default=1.0e-6)
             Error tolerance used to check convergence in power method iteration.

           nstart : dictionary, optional (default=None)
             Starting value of eigenvector iteration for each node.

           weight : None or string, optional (default=None)
             If None, all edge weights are considered equal.
             Otherwise holds the name of the edge attribute used as weight.

           Returns
           -------
           nodes : dictionary
              Dictionary of nodes with eigenvector centrality as the value.

           Examples
           --------
           >>> G = nx.path_graph(4)
           >>> centrality = nx.eigenvector_centrality(G)
           >>> sorted((v, '{:0.2f}'.format(c)) for v, c in centrality.items())
           [(0, '0.37'), (1, '0.60'), (2, '0.60'), (3, '0.37')]

           Raises
           ------
           NetworkXPointlessConcept
               If the graph `G` is the null graph.

           NetworkXError
               If each value in `nstart` is zero.

           PowerIterationFailedConvergence
               If the algorithm fails to converge to the specified tolerance
               within the specified number of iterations of the power iteration
               method.

           See Also
           --------
           eigenvector_centrality_numpy
           pagerank
           hits

           Notes
           -----
           The measure was introduced by [1]_ and is discussed in [2]_.

           The power iteration method is used to compute the eigenvector and
           convergence is **not** guaranteed. Our method stops after ``max_iter``
           iterations or when the change in the computed vector between two
           iterations is smaller than an error tolerance of
           ``G.number_of_nodes() * tol``. This implementation uses ($A + I$)
           rather than the adjacency matrix $A$ because it shifts the spectrum
           to enable discerning the correct eigenvector even for networks with
           multiple dominant eigenvalues.

           For directed graphs this is "left" eigenvector centrality which corresponds
           to the in-edges in the graph. For out-edges eigenvector centrality
           first reverse the graph with ``G.reverse()``.

           References
           ----------
           .. [1] Phillip Bonacich.
              "Power and Centrality: A Family of Measures."
              *American Journal of Sociology* 92(5):1170–1182, 1986
              <http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf>
           .. [2] Mark E. J. Newman.
              *Networks: An Introduction.*
              Oxford University Press, USA, 2010, pp. 169.

           """

        try:
            self.logger.info('正在计算网络的特征向量中心性 ...')
            return self.order_dict(nx.eigenvector_centrality(self.G), index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    # 同配系数 ---------------------------------------------------------------------------------------------------
    def degree_assortativity_coefficient(self, x='in', y='in', **kwargs):
        """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    Parameters
    ----------
    G : NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Compute degree assortativity only for nodes in container.
        The default is all nodes.

    Returns
    -------
    r : float
       Assortativity of graph by degree.

    Examples
    --------
    >>> G=nx.path_graph(4)
    >>> r=nx.degree_assortativity_coefficient(G)
    >>> print("%3.1f"%r)
    -0.5

    See Also
    --------
    attribute_assortativity_coefficient
    numeric_assortativity_coefficient
    neighbor_connectivity
    degree_mixing_dict
    degree_mixing_matrix

    Notes
    -----
    This computes Eq. (21) in Ref. [1]_ , where e is the joint
    probability distribution (mixing matrix) of the degrees.  If G is
    directed than the matrix e is the joint probability of the
    user-specified degree type for the source and target.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M.
       Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).
    """
        try:
            self.logger.info('正在计算网络的同配系数 ...')
            if kwargs.get('pearson'):
                return nx.degree_pearson_correlation_coefficient(self.G, x=x, y=y, weight=kwargs.get('weight'),
                                                                 nodes=kwargs.get('nodes'))
            else:
                return nx.degree_assortativity_coefficient(self.G, x=x, y=y, weight=kwargs.get('weight'),
                                                           nodes=kwargs.get('nodes'))
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))


# 科研指标
class ResMetrics():

    # h-index ----------------------------------------------------------------------------------------------------
    def h_index(self):
        """
        h-index ，又称为h指数或h因子（h-factor），是一种评价学术成就的新方法。
        h代表“高引用次数”（high citations），一名科研人员的h指数是指他至多有h篇论文分别被引用了至少h次。
        h指数能够比较准确地反映一个人的学术成就。一个人的h指数越高，则表明他的论文影响力越大。
        例如，某人的h指数是20，这表示他已发表的论文中，每篇被引用了至少20次的论文总共有20篇。
        要确定一个人的h指数非常容易，到SCI网站，查出某个人发表的所有SCI论文，让其按被引次数从高到低排列，往下核对，直到某篇论文的序号大于该论文被引次数，那个序号减去1就是h指数。
        中国读者较为熟悉的霍金的h指数比较高，为62。
        生物学家当中h指数最高的为沃尔夫医学奖获得者、约翰斯·霍普金斯大学神经生物学家施奈德，高达191，
        其次为诺贝尔生理学或医学奖获得者、加州理工学院生物学家巴尔的摩，160。
        生物学家的h指数都偏高，表明h指数就像其他指标一样，不适合用于跨学科的比较。

        计算网络节点的H-index
        思路：考虑节点的度属性和邻居关系，即：
            一个节点的h-index为k，表示至少有k个邻居的度不小于k，有向网络仅考虑入度
        :return: dict
        """

        def hindex(ls):
            if ls:
                ls.sort()  # 排序算法 最耗时的部分
                h = 1 if ls[-1] > 0 else 0
                small = ls[-1]
                for i in ls[-2::-1]:
                    if i == small and i > h:
                        h += 1
                    elif i > h:
                        h += 1
                        small = i
                    else:
                        break
                return h
            else:
                return 0

        try:
            degrees = dict(self.in_degree() if self.G.is_directed() else self.degree())
            neighbors = self.in_neighbor() if self.G.is_directed() else self.neighbor()
            h_index = {node: hindex([degrees[i] for i in neighbors[node]]) for node in self.G}
            return self.order_dict(h_index, index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))

    # CI ---------------------------------------------------------------------------------------------------------
    def CI(self):
        """
        计算网络传播指标CI: collective influence

        The idea is to determine the nodes with the most influence on a network
        that are capable of collapsing or totally fragmenting the network when they are removed.
        After the first iteration, the CI value is calculated and sorted in descending order.
        The node with the highest CI value is removed and the calculation is repeated and CI sorted again
        to determine the node with the next highest CI value.
        The cycle continues until the network is completely fragmented with the least number of largest components.

        计算公式：CI(i)=(ki - 1) * sum([kj -1 for j in neighbor(i)])
        :return: CI
        """
        neighbors_info = self.neighbor()
        degree_info = self.degree()
        CI_info = {node: 0 for node in self.G}
        for node in self.G:
            if degree_info[node] > 1:
                CI_info[node] = (degree_info[node] - 1) * sum(
                    [(degree_info[n] - 1) if degree_info[n] > 1 else 0 for n in neighbors_info[node]])
        return self.order_dict(CI_info, index=1)

    def CI_in(self):
        neighbors_info = self.in_neighbor()
        degree_info = self.in_degree()
        CI_info = {node: 0 for node in self.G}
        for node in self.G:
            if degree_info[node] > 1:
                CI_info[node] = (degree_info[node] - 1) * sum(
                    [(degree_info[n] - 1) if degree_info[n] > 1 else 0 for n in neighbors_info[node]])
        return self.order_dict(CI_info, index=1)

    def CI_out(self):
        neighbors_info = self.out_neighbor()
        degree_info = self.in_degree()
        CI_info = {node: 0 for node in self.G}
        for node in self.G:
            if degree_info[node] > 1:
                CI_info[node] = (degree_info[node] - 1) * sum(
                    [(degree_info[n] - 1) if degree_info[n] > 1 else 0 for n in neighbors_info[node]])
        return self.order_dict(CI_info, index=1)

    # 核数 -------------------------------------------------------------------------------------------------------
    def core_num(self):
        """Returns the core number for each vertex.

        A k-core is a maximal subgraph that contains nodes of degree k or more.

        The core number of a node is the largest value k of a k-core containing
        that node.

        Parameters
        ----------
        G : NetworkX graph
           A graph or directed graph

        Returns
        -------
        core_number : dictionary
           A dictionary keyed by node to the core number.

        Raises
        ------
        NetworkXError
            The k-core is not implemented for graphs with self loops
            or parallel edges.

        Notes
        -----
        Not implemented for graphs with parallel edges or self loops.

        For directed graphs the node degree is defined to be the
        in-degree + out-degree.

        References
        ----------
        .. [1] An O(m) Algorithm for Cores Decomposition of Networks
           Vladimir Batagelj and Matjaz Zaversnik, 2003.
           https://arxiv.org/abs/cs.DS/0310049
        """
        try:
            self.logger.info('正在计算简单图的核数值 ...')
            return self.order_dict(nx.core_number(self.G), index=1)
        except Exception as e:
            self.logger.error("计算失败，原因：{0}".format(e))
            return {}

    def in_core_num(self):

        degrees = dict(self.in_degree())
        # Sort nodes by degree.
        nodes = sorted(degrees, key=degrees.get)
        bin_boundaries = [0]
        curr_degree = 0
        for i, v in enumerate(nodes):
            if degrees[v] > curr_degree:
                bin_boundaries.extend([i] * (degrees[v] - curr_degree))
                curr_degree = degrees[v]
        node_pos = {v: pos for pos, v in enumerate(nodes)}
        # The initial guess for the core number of a node is its degree.
        core = degrees
        # in_nbrs = self.in_neighbor()
        in_nbrs = self.out_neighbor()
        nbrs = {v: list(in_nbrs.get(v)) for v in in_nbrs}
        for v in nodes:
            for u in nbrs[v]:
                if core[u] > core[v]:
                    nbrs[v].remove(u)
                    pos = node_pos[u]
                    bin_start = bin_boundaries[core[u]]
                    node_pos[u] = bin_start
                    node_pos[nodes[bin_start]] = pos
                    nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                    bin_boundaries[core[u]] += 1
                    core[u] -= 1
        return self.order_dict(core, index=1)

    def out_core_num(self):
        degrees = dict(self.out_degree())
        # Sort nodes by degree.
        nodes = sorted(degrees, key=degrees.get)
        bin_boundaries = [0]
        curr_degree = 0
        for i, v in enumerate(nodes):
            if degrees[v] > curr_degree:
                bin_boundaries.extend([i] * (degrees[v] - curr_degree))
                curr_degree = degrees[v]
        node_pos = {v: pos for pos, v in enumerate(nodes)}
        # The initial guess for the core number of a node is its degree.
        core = degrees
        out_nbrs = self.in_neighbor()
        nbrs = {v: list(out_nbrs.get(v)) for v in out_nbrs}
        for v in nodes:
            for u in nbrs[v]:
                if core[u] > core[v]:
                    nbrs[v].remove(u)
                    pos = node_pos[u]
                    bin_start = bin_boundaries[core[u]]
                    node_pos[u] = bin_start
                    node_pos[nodes[bin_start]] = pos
                    nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                    bin_boundaries[core[u]] += 1
                    core[u] -= 1
        return self.order_dict(core, index=1)

    def dhc(self, x, y):
        """
        作图法求解节点的dhc值
        :param x: edge weight list
        :param y: dhc list
        :return: dhc, float

        piecewise_function：分段函数
        """

        def piecewise_function(x, y, x0):
            """
            分段函数
            :param x: 交点x坐标，list
            :param y: 交点y坐标，list
            :param x0: 目标点x坐标，float
            :return: 目标点y坐标
            注：默认首个焦点为与y轴交点，且不在x中
            """
            for i in range(len(x)):
                if x0 <= x[i]:
                    return y[i]
            return 0

        if 0 in y:  # 邻居的dhc值为零时，舍去
            for index, item in enumerate(y):
                if item == 0:
                    y.pop(index)
                    x.pop(index)

        if not x:  # 邻居的dhc全为0时，返回0
            return 0

        x = [sum(x[:i]) for i in range(1, len(x) + 1)]

        left = 0
        right = x[-1] + 0.01
        left_value = piecewise_function(y=y, x=x, x0=left) - left
        right_value = piecewise_function(y=y, x=x, x0=right) - right
        EPS = right / 10000
        while right - left >= EPS:  # 二分法求零点
            center = (right + left) / 2
            center_value = piecewise_function(y=y, x=x, x0=center) - center
            if center_value == 0:
                return center
            elif left_value == 0:
                return left
            elif right_value == 0:
                return right
            elif left_value * center_value < 0:
                right = center
                right_value = piecewise_function(y=y, x=x, x0=right) - right
            elif right_value * center_value < 0:
                left = center
                left_value = piecewise_function(y=y, x=x, x0=left) - left
        # return round(center, 5)
        return center

    def weighted_core_num(self, nbrs, dhcs):
        for i in range(self.ITER):
            node = random.choice(list(dhcs))
            neighbors = nbrs.get(node)
            if neighbors:
                x = [nbr['weight'] for nbr in neighbors.values()]
                y = [dhcs.get(nbr) for nbr in neighbors]
                data = {x[i]: y[i] for i in range(len(x))}
                data = sorted(data.items(), key=lambda d: d[1], reverse=True)
                x = [item[0] for item in data]
                y = [item[1] for item in data]
                dhcs[node] = self.dhc(x, y)
            else:
                dhcs[node] = 0
        return self.order_dict(dhcs, index=1)

    def core(self, weighted=False):
        """
        计算无向网络核数
        :param weighted: bool
        :return: dict
        """
        if weighted:
            return self.order_dict(self.weighted_core_num(self.neighbor(), self.strength()), index=1)
        else:
            return self.order_dict(self.core_num(), index=1)

    def in_core(self, weighted=False):
        """
        计算有向网络入核数
        :param weighted: bool
        :return: dict
        """
        if weighted:
            return self.order_dict(self.weighted_core_num(self.in_neighbor(), self.in_strength()), index=1)
        else:
            return self.order_dict(self.in_core_num(), index=1)

    def out_core(self, weighted=False):
        """
        计算有向网络出核数
        :param weighted: bool
        :return: dict
        """
        if weighted:
            return self.order_dict(self.weighted_core_num(self.out_neighbor(), self.out_strength(hide_log=True)), index=1)
        else:
            return self.order_dict(self.out_core_num(), index=1)


class NetworkxPython(Network, LibMetrics, ResMetrics, LoggingPython, Common):

    def __init__(self, **kwargs):
        LoggingPython.__init__(self, log_name='networkx')
        self.G = self.network()
        self.ITER = 10000

    def data(self):
        return []

    def network(self, **kwargs):
        G = self.er(n=20, p=0.2)  # 随机网络
        return G

    def job(self):
        pass

    def show(self):
        import matplotlib.pyplot as plt
        pos = nx.shell_layout(self.G)
        # pos = nx.spring_layout(self.G)
        width = [self.G.edges().get(edge).get('weight', 1) for edge in self.G.edges()]
        nx.draw_networkx(self.G, pos=pos, width=width)
        plt.show()


class Successor(NetworkxPython):
    def __init__(self, **kwargs):
        NetworkxPython.__init__(self)
        self.G = self.network()

    def network(self):
        data = self.data()
        # 图对象 ---------------------------------------------------------------------------------------------------
        if data:
            # G = self.digraph(data, is_weighted=False)  # 导入数据，生成有向（含权）网络
            G = self.digraph(data, is_weighted=True)  # 导入数据，生成有向（含权）网络
            # G = self.graph(data, is_weighted=False)  # 导入数据，生成无向（含权）网络
            # G = self.graph(data, is_weighted=True)  # 导入数据，生成无向（含权）网络
        else:
            # G = self.rg(d=3, n=20)  # 规则网络
            G = self.er(n=20, p=0.2)  # 随机网络
            # G = self.er(n=20, p=0.1, directed=True)  # 随机网络
            # G = self.ws(n=20, k=4, p=0.3)  # 小世界网络
            # G = self.ba(n=20, m=2)  # 无标度网络
        return G

    def job(self):
        self.show()
        # print(self.is_directed())  # 判断是否为有向图
        # print(self.neighbor())  # 返回节点邻居
        # print(self.in_neighbor())  # 返回节点入邻居
        # print(self.out_neighbor())  # 返回节点出邻居
        # print(list(self.connected_components()))  # 默认返回所有连通片，max=True，则返回最大连通片

        # print(self.degree())  # 返回节点度
        # print(self.in_degree())  # 返回节点入度
        # print(self.out_degree())  # 返回节点出度
        # print(self.strength())  # 返回节点强度
        # print(self.in_strength())  # 返回节点入强度
        # print(self.out_strength())  # 返回节点出强度
        # print(self.pagerank(alpha=0.85))  # 返回节点的PR值
        # print(self.clustering())  # 返回节点的聚类系数
        # print(self.average_clustering())  # 计算网络的平均聚类系数
        # print(self.degree_centrality())  # 返回节点度中心性
        # print(self.in_degree_centrality())  # 返回节点入度中心性
        # print(self.out_degree_centrality())  # 返回节点出度中心性
        # print(self.closeness_centrality())  # 返回节点接近中心性
        # print(self.betweenness_centrality())  # 返回节点介数中心性
        # print(self.eigenvector_centrality())  # 返回节点特征向量中心性
        # print(self.degree_assortativity_coefficient())  # 计算网络的同配系数

        # print(self.h_index())  # 返回节点的h-index值
        # print(self.CI())  # 返回节点的CI值
        # print(self.CI_in())  # 返回节点的CI_in值
        # print(self.CI_out())  # 返回节点的CI_out值
        # print(self.core())  # 返回节点的核数值
        # print(self.in_core())  # 返回节点的入向核数值
        # print(self.out_core())  # 返回节点的出向核数值
        # print(self.core(weighted=True))  # 返回节点的含权核数值
        # print(self.in_core(weighted=True))  # 返回节点的入向含权核数值
        # print(self.out_core(weighted=True))  # 返回节点的出向含权核数值


if __name__ == '__main__':
    data = []
    # data = [('A', 'B', 1), ('B', 'C', 2), ('C', 'A', 3), ('D', 'C', 4)]
    data = [('A', 'B', 1), ('B', 'C', 2), ('C', 'A', 3), ('C', 'D', 4)]
    client = Successor(data=data)
    client.job()
    pass
