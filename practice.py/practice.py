import tensorflow as tf

x = [1,2,3]
y = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))
H = W*x+b
cost = tf.reduce_mean(tf.square(H-y))

a = tf.Variable(0.1)
opt = tf.train.GradientDescentOptimizer(a)
train = opt.minimize(cost)

init = tf.initialize_all_variables()

sess= tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step%20 ==0:
        print step, sess.run(cost), sess.run(W), sess.run(b)