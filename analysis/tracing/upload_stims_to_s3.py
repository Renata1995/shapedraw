import os
import boto
import numpy as np

def deborkify_imlist(im_list):
	im_list = [i for i in im_list if i != '.DS_Store']
	im_list = [i for i in im_list if i[-3:]=='png']
	return im_list

def construct_full_paths(im_list,
						stims_dir = '../tracing_eval_stims',
						this_exp = 'museumstation_v3',
						this_shape = 'circle'):
	base_path = os.path.join(stims_dir,this_exp,this_shape)
	path_list = [os.path.join(base_path,this_im) for this_im in im_list]
	return path_list

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--bucket_name', type=str, help='name of S3 bucket?', \
		default='kiddraw-tracing')
	parser.add_argument('--path_to_imgs', type=str, help='path to images to upload?', \
		default='./tracing_eval_stims')   
	args = parser.parse_args()

	## tell user some useful information
	print 'Path to images is : {}'.format(args.path_to_imgs)    
	print 'Uploading to this bucket: {}'.format(args.bucket_name)

	## establish connection to s3 
	conn = boto.connect_s3()

	## create a bucket with the appropriate bucket name
	try:
		b = conn.create_bucket(args.bucket_name) 
	except:
		b = conn.get_bucket(args.bucket_name) 

	## path to stims
	path_to_imgs = args.path_to_imgs	

	## define paths to experiment subdirs within stim dir
	_experiment_list = os.listdir(args.path_to_imgs)	
	experiment_list = [i for i in _experiment_list if i != '.DS_Store'] ## mac nonsense

	## define paths to shape subdirs within experiment dirs
	for exp_ind,this_exp in enumerate(experiment_list):	
		shape_list = os.listdir(os.path.join(args.path_to_imgs,this_exp))
		shape_list = [i for i in shape_list if i != '.DS_Store']
		for shape_ind, this_shape in enumerate(shape_list):
			im_list = os.listdir(os.path.join(args.path_to_imgs,this_exp,this_shape))
			im_list = deborkify_imlist(im_list) 			
			path_to_ims = construct_full_paths(im_list,
												stims_dir=path_to_imgs,
												this_exp=this_exp,
												this_shape=this_shape)
			for im_ind, this_im in enumerate(path_to_ims):
				im_name = this_im.split('/')[-1]
				try:
					k = b.new_key(im_name) ## if we need to overwrite this, we have to replace this line boto.get_key
				except:
					k = b.get_key(im_name)
				k.set_contents_from_filename(this_im)
				k.set_acl('public-read')
				print 'Uploading {} of {} | {}'.format(im_ind,len(path_to_ims),im_name)
