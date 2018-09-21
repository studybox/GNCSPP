import tensorflow as tf
import tensorflow.contrib.layers as layers

#TODO Verify this part of the code need graph building
def _struct_2_vec(hidden, embeditrn, numnodes, numedges, numglobals, inpt, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        # The input is of the format (b, 2n+4e+u), for now let's go with dense matrices
        n1,n2,e1,e2,e3,e4,e5,u = tf.split(inpt, [numnodes,numnodes,numedges,numedges,numedges,numedges,numedges,numglobal], axis=1)
        # n1 is of shape (b, n, 1)
        n1 = tf.expand_dims(n1, axis=2)
        #nodelinears is of shape (b, n, p)
        nodelinears = layers.fully_connected(n1, num_outputs=hidden, activation_fn=None, biases_initializer=None)
        # u is of shape (b, 1, u)
        u = tf.expand_dims(u, axis=1)
        # nu is of shape (b, n, u)
        nu = tf.tile(u, [1, numnodes, 1])
        #globallinears is of shape (b, n, p)
        globallinears = layers.fully_connected(nu, num_outputs=hidden, activation_fn=None, biases_initializer=None)
        # e1e2 is of shape (b, e, 2)
        e1e2 = tf.stack([e1, e2], axis=2)
        #edgelinears is of shape (b, e, p)
        edgelinears = layers.fully_connected(e1e2, num_outputs=hidden, activation_fn=tf.nn.relu)

        # outnodes is a list length n of out nodes
        outnodes = tf.split(e5, n2, axis = 1)
        # khotsnode is a list of indicator vectors
        khotsnode = []
        for on in outnodes:
            khotsnode.append(tf.reduce(tf.one_hot(on, numnodes), axis=1))
        # khotsnode is of shape (b, n, n)
        khotsnode = tf.stack(khotsnode, axis = 1)

        # outedges is a list length n of out edges
        outedges = tf.split(e4, n2, axis = 1)
        # khotsedge is a list of indicator vectors
        khotsedge = []
        for oe in outedges:
            khotsedge.append(tf.reduce(tf.one_hot(oe, numedges), axis=1))
        # khots is of shape (b, n, e)
        khotsedge = tf.stack(khotsedge, axis = 1)
        # edgesums is of shape (b, n, p)
        edgesums = tf.matmul(knotsedge, edgelinears)

        # edgesumlinears is of shape (b, n, p)
        edgesumlinears = layers.fully_connected(edgesums, num_outputs=hidden, activation_fn=None)
        # inptlinears is of shape (b, n, p)
        inptlinears = nodelinears + edgesumlinears + globallinears
        curinputlinears = tf.nn.relu(inputlinears) # (b, n, p)
        #W = tf.get_variable("graph_W", [hidden, hidden], initializer=tf.initializers.xavier_initializer())

        for level in range(embeditrn):
            #outlinears = curinputlinears @ W
            outlinears = layers.fully_connected(curinputlinears, num_outputs=hidden, activation_fn=None, reuse=True, scope="graph") #(b, n, p)
            #neighborsums is of shape (b, n, p)
            neighborsums = tf.matmul(knotsnode, outlinears)
            # curinputlinears is of shape (b, n, p)
            curinputlinears = tf.nn.relu(neighborsums + inptlinears)
        nodeembed = curinputlinears
        # insert a second GN block
        e4perm = tf.one_hot(e4, numedges) # (b, e, e)
        e3 = tf.expand_dims(e3, axis= 2) # (b, e, 1)
        e3perm = tf.matmul(e4perm, e3) # (b, e, 1)
        zero = tf.constant(0, dtype=e3.dtype)
        one = tf.constant(1, dtype=e3.dtype)
        #pathindexs = tf.where(tf.not_equal(e3perm, zero))
        pathedge =  e3perm # of shape (b, e, 1)
        #tf.gather(e3perm, pathindexs) # of shape (b, k, 1)

        e3linears = layers.fully_connected(pathedge, num_outputs=hidden, activation_fn=None, biases_initializer=None) # of shape(b, e, p)
        e3linears = tf.expand_dims(e3linears, axis=3) # of shape (b, e, p, 1)
        pathMat = layers.fully_connected(e3linears, num_outputs=hidden, activation_fn=None, biases_initializer=None) #of shape (b, e, p, p)

        e5perm = tf.one_hot(e5, numnodes) # of shape (b, e, n)
        nodeembedsperm = tf.expand_dims(tf.matmul(e5perm, nodeembed), axis = 3) # of shape (b, e, p, 1)
        edgeembed = tf.matmul(pathMat, nodeembedsperm) # of shape (b, e, p, 1)
        edgeembed = tf.squeeze(edgeembed, axis=3) # of shape (b, e, p)
        # actions
        # unstack path
        pathlist = tf.unstack(tf.squeeze(pathedge, axis=2), axis=0) # b length list of (e, )
        actidx = []
        # unstack edgeembed
        edgeembedlist = tf.unstack(edgeembed, axis=0) # b length list of (e,p)
        # qlinears should be (b,1,p)
        qlinears = layers.fully_connected(tf.reduce_sum(edgeembed, axis=1, keepdims=True), num_outputs=hidden, activation_fn=tf.nn.relu)
        qlist = tf.unstack(qlinears, axis=0) # b length list of (1, p)
        qout = []
        for idx, p in enumerate(pathlist):
            actionindex = tf.where(tf.less(p, one))
            actidx.append(actionindex)
            actembeds = tf.gather(edgeembedlist[idx], actionindex) # (k, p)
            actlinears = layers.fully_connected(actembeds, num_outputs=hidden, activation_fn=tf.nn.relu, reuse=True, scope="action") # (k, p)
            qklinears = tf.tile(qlist[idx], [actionindex.shape[0],1])  # (k, p)
            q = tf.concat([qklinears, actlinears], axis=1)
            qout.append(q)
        return qout, actidx
def struct_2_vec(hidden, embeditrn, numnodes, numedges, numglobals):
    """
    The model is adopted based on structure_2_vec in the paper Dai. et.al, ICML 2016,
    The Q function is defined as:
        Q = theta_5*relu([theta_6*sum(miu_u), miu_v])
        miu_v = relu(theta_1*x_v + theta_2*sum(p_u) + theta_3*sum(relu(theta_4*w(v,u))))
    Parameters
    ----------
    hidden: int
        the dimension of the embedded node code
    embeditrn: int
        number of iteration of embedding process
    Returns
    -------
    q_func: function
        q_function for Q learning
    """
    return lambda *args, **kwargs: _struct_2_vec(hidden, embeditrn, numnodes, numedges, numglobals, *args, **kwargs)
