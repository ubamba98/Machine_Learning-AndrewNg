import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

node1 = 500
node2 = 500
node3 = 500
classes = 10

batch_size = 100
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')
def NNM(data):  
    h1l = {'weights':tf.Variable(tf.random_normal([784,node1])),
          'biases':tf.Variable(tf.random_normal([node1]))}
    h2l = {'weights':tf.Variable(tf.random_normal([node1,node2])),
          'biases':tf.Variable(tf.random_normal([node2]))}
    h3l = {'weights':tf.Variable(tf.random_normal([node2,node3])),
          'biases':tf.Variable(tf.random_normal([node3]))} 
    ol = {'weights':tf.Variable(tf.random_normal([node3,classes])),
          'biases':tf.Variable(tf.random_normal([classes]))}
    
    l1 = tf.add(tf.matmul(data,h1l['weights']),h1l['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1 ,h2l['weights']),h2l['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,h3l['weights']),h3l['biases'])
    l3 = tf.nn.relu(l3) 
    oll = tf.add(tf.matmul(l3,ol['weights']),ol['biases'])
    
    return oll

def train(x):
    prediction = NNM(x)
    cost =  tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_cost = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epx,epy = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict = {x: epx, y: epy})
                epoch_cost +=c
            print('Epoch', epoch,'completed out of',hm_epochs,'loss:',epoch_cost)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        acc = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',acc.eval({x:mnist.test.images,y:mnist.test.labels}))
train(x)        
        
        