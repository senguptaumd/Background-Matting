import numpy as np
import cv2, pdb, glob, argparse

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2,masksDL):

	# Convert images to grayscale
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	akaze = cv2.AKAZE_create()
	keypoints1, descriptors1 = akaze.detectAndCompute(im1, None)
	keypoints2, descriptors2 = akaze.detectAndCompute(im2, None)
	
	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
	matches = matcher.match(descriptors1, descriptors2, None)
	
	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False)

	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]
	
	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt
	
	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# Use homography
	height, width, channels = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))
	# copy image in the empty region, unless it is a foreground. Then copy background

	mask_rep=(np.sum(im1Reg.astype('float32'),axis=2)==0)

	im1Reg[mask_rep,0]=im2[mask_rep,0]
	im1Reg[mask_rep,1]=im2[mask_rep,1]
	im1Reg[mask_rep,2]=im2[mask_rep,2]

	mask_rep1=np.logical_and(mask_rep , masksDL[...,0]==255)

	im1Reg[mask_rep1,0]=im1[mask_rep1,0]
	im1Reg[mask_rep1,1]=im1[mask_rep1,1]
	im1Reg[mask_rep1,2]=im1[mask_rep1,2]


	return im1Reg

def adjustExposure(img,back,mask):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	mask = cv2.dilate(mask, kernel, iterations=10)
	mask1 = cv2.dilate(mask, kernel, iterations=300)
	msk=mask1.astype(np.float32)/255-mask.astype(np.float32)/255; msk=msk.astype(np.bool)

	bias=np.zeros((1,3)); gain=np.ones((1,3))

	bias[0,0],gain[0,0]=bias_gain(img[...,0],back[...,0],msk)
	bias[0,1],gain[0,1]=bias_gain(img[...,1],back[...,1],msk)
	bias[0,2],gain[0,2]=bias_gain(img[...,2],back[...,2],msk)

	return bias,gain


def bias_gain(orgR,capR,cap_mask):

	xR=capR[cap_mask]
	yR=orgR[cap_mask]

	gainR=np.nanstd(yR)/np.nanstd(xR);
	biasR=np.nanmean(yR)-gainR*np.nanmean(xR);

	return biasR,gainR


parser = argparse.ArgumentParser(description='Deeplab Segmentation')
parser.add_argument('-i', '--input_dir', type=str, required=True,help='Directory to save the output results. (required)')
parser.add_argument('-v_name','--video_name',type=str, default=None,help='Name of the video')
args=parser.parse_args()


dir_name=args.input_dir

list_im=glob.glob(dir_name + '/*_img.png'); list_im.sort()


back=cv2.imread(args.video_name);


# back=back.astype('float32')/255
# #adjust bias-gain
# bias=[]; gain=[]
# for i in range(0,len(list_im),30):

# 	image = cv2.imread(list_im[i]); image=image.astype('float32')/255
# 	mask = cv2.imread(list_im[i].replace('img','masksDL'))

# 	b,g=adjustExposure(image,back,mask[...,0])
# 	bias.append(b); gain.append(g)
# Bias=np.median(np.asarray(bias),axis=0).squeeze(0); 
# Gain=np.median(np.asarray(gain),axis=0).squeeze(0)
# back_new=back
# back_new[...,0]=Gain[0]*back[...,0]+Bias[0]
# back_new[...,1]=Gain[1]*back[...,1]+Bias[1]
# back_new[...,2]=Gain[2]*back[...,2]+Bias[2]
# back_new=(255*back_new).astype(np.uint8)

for i in range(0,len(list_im)):

	image = cv2.imread(list_im[i])
	mask = cv2.imread(list_im[i].replace('img','masksDL'))

	back_align = alignImages(back, image,mask)

	cv2.imwrite(list_im[i].replace('img','back'),back_align)

	print('Done: ' + str(i+1) + '/' + str(len(list_im)))


	
