import tensorflow as tf

x_data=[1.0,2.0,3.0]
x2_data = [0,2,0,4,]
y_data=[1.0,2.0,3.0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))
H = W*X+b
cost = tf.reduce_mean(tf.square(H-Y))

a = tf.Variable(0.1)
opt = tf.train.GradientDescentOptimizer(a)
train = opt.minimize(cost)

init = tf.initialize_all_variables()

sess= tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data,Y:y_data}), sess.run(W), sess.run(b)

print sess.run(H,feed_dict={X:5})
print sess.run(H,feed_dict={X:2.6})
