import numpy as np 
import tensorflow.compat.v1 as tf
from tensorflow.io.gfile import GFile
tf.disable_v2_behavior()

with tf.Session() as sess:	    
	with GFile('../models/model.pb', 'rb') as f:	        
		graph_def = tf.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	     
		g_in = tf.import_graph_def(graph_def)	  
		print([n.name for n in tf.get_default_graph().as_graph_def().node])     
		tensor_input = sess.graph.get_tensor_by_name('import/Placeholder_2:0')	        
		tensor_output = sess.graph.get_tensor_by_name('import/dense_3/BiasAdd:0')	        	                    	   



test_sample = np.array([[14, .4,.0, 1., 1.]])
with tf.Session() as sess:
  predictions = sess.run(tensor_output, {tensor_input:test_sample})	        


print(predictions)

