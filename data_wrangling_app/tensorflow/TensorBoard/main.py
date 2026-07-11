import tensorflow as tf
import numpy as np

raw_data = np.random.normal(loc=10, scale=1, size=100)

#The moving average is defined as follows
alpha = tf.constant(value=0.05)

curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(initial_value=0.)

update_avg = alpha * curr_value + (1 - alpha) * prev_avg

#Heres what we care to to visualize
avg_hist = tf.summary.scalar(name="running_average", data=update_avg)

value_hist = tf.summary.scalar(name="incoming_values", data=curr_value)

merged = tf.summary.merge_all()
writer = tf.summary.FileWrite("./logs")

#compute the moving averges, also run the merged op to track how the values change
init = tf.global_variables_intializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(raw_data)):
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value: raw_data[i]})
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
        writer.add_summary(summary_str, i)



