import tensorflow as tf
import tensorflow.contrib.layers as layers

#TODO Verify this part of the code need graph building
def _struct_2_vec(hidden, embeditrn, numbatchs,  numnodes, numedges, numglobals, outnodes, outedges, edgev, inpt, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        # The input is of the format (b, 2n+5e+u), for now let's go with dense matrices
        n1,e1,e2,e3,e4,u = tf.split(inpt, [numnodes,numedges,numedges,numedges,numedges,numglobals], axis=1)
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


        # khotsnode is a list of indicator vectors
        khotsnode = []
        for on in outnodes:
            khotsnode.append(tf.reduce(tf.one_hot(on, numnodes), axis=0))
        # khotsnode is of shape (b, n, n)
        khotsnode = tf.tile(tf.expand_dims(tf.stack(khotsnode, axis=0), axis=0), [n1.shape[0],1,1])


        # khotsedge is a list of indicator vectors
        khotsedge = []
        for oe in outedges:
            khotsedge.append(tf.reduce(tf.one_hot(oe, numedges), axis=0))
        # khotedges is of shape (b, n, e)
        khotsedge = tf.tile(tf.expand_dims(tf.stack(khotsedge, axis=0), axis=0), [n1.shape[0],1,1])
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
        nodeembed = curinputlinears #(b, n, p)
        # insert a second GN block
        receivers = tf.tile(tf.expand_dims(tf.one_hot(edgev, numnodes), axis=0), [n1.shape[0],1,1]) # (b, e, n)
        pathedge = tf.expand_dims(e3, axis= 2) # (b, e, 1)
        #tf.gather(e3perm, pathindexs) # of shape (b, k, 1)
        zero = tf.constant(0, dtype=e3.dtype)
        e3linears = layers.fully_connected(pathedge, num_outputs=hidden, activation_fn=None, biases_initializer=None) # of shape(b, e, p)
        e3linears = tf.expand_dims(e3linears, axis=3) # of shape (b, e, p, 1)
        pathMat = layers.fully_connected(e3linears, num_outputs=hidden, activation_fn=None, biases_initializer=None) #of shape (b, e, p, p)

        nodeembedsperm = tf.expand_dims(tf.matmul(receivers, nodeembed), axis = 3) # of shape (b, e, p, 1)
        edgeembed = tf.matmul(pathMat, nodeembedsperm) # of shape (b, e, p, 1)
        edgeembed = tf.squeeze(edgeembed, axis=3) # of shape (b, e, p)
        # actions (b, e, 1)
        actions = tf.expand_dims(e4, axis=2)
        # unstack path
        actlinears = actions * edgeembed # shape(b, e, p)
        actionlist = tf.unstack(e4, axis=0) # b length list of (e, )
        actidx = []
        # unstack edgeembed
        edgeembedlist = tf.unstack(edgeembed, axis=0) # b length list of (e,p)
        # qlinears should be (b,1,p)
        qlinears = layers.fully_connected(tf.reduce_sum(edgeembed, axis=1, keepdims=True), num_outputs=hidden, activation_fn=tf.nn.relu)
        qnlinears = tf.tile(qlist[idx], [1, numedges,1])  # (b, e, p)
        qnlinears2 = tf.concat([qnlinears, actlinears], axis=2) #(b, e, 2p)
        q = tf.squeeze(layers.fully_connected(qnlinears2, num_outputs=1, activation_fn=None, biases_initializer=None), [2]) # (b, e)

        qlist = tf.unstack(q, axis=0) # b length list of (e, )
        qout = []
        for idx, a in enumerate(actionlist):
            actionindex = tf.reshape(tf.where(tf.not_equal(a, zero)), [-1])
            actidx.append(actionindex)
            #actembeds = tf.gather(edgeembedlist[idx], actionindex) # (k, p)
            #actlinears = layers.fully_connected(actembeds, num_outputs=hidden, activation_fn=tf.nn.relu, reuse=True, scope="action") # (k, p)
            #qklinears = tf.tile(qlist[idx], [actionindex.shape[0],1])  # (k, p)
            #q = tf.concat([qklinears, actlinears], axis=1)
            qout.append(tf.gather(qlist[idx], actionindex))
        return qout, actidx
def struct_2_vec(hidden, embeditrn, numbatchs, numnodes, numedges, numglobals, outdegrees, outedges, outnodes, edgev):
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
    # outnodes is a list length n of out nodes
    outnodes = np.split(outnodes, outdegrees, axis = 1)
    # outedges is a list length n of out edges
    outedges = np.split(outedges, outdegrees, axis = 1)
    return lambda *args, **kwargs: _struct_2_vec(hidden, embeditrn, numbatchs, numnodes, numedges, numglobals, outnodes, outedges, edgev, *args, **kwargs)
