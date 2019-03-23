import tensorflow as tf



















#
# a = tf.get_variable('a', [1], initializer=tf.constant_initializer(1.0))
# print(a.name)
# with tf.variable_scope('one'):
#     a2 = tf.get_variable('a', [1], initializer=tf.constant_initializer(1.0))
#     print(a2.name)
#
# with tf.variable_scope('one'):
#     with tf.variable_scope('two'):
#         a4 = tf.get_variable('a', [1])
#         print(a4.name)
#     b = tf.get_variable('b', [1])
#     print(b.name)
#
# with tf.variable_scope('', reuse=True):
#     a5 = tf.get_variable('one/two/a', [1])
#     print(a5 == a4)
