import co
import numpy as np
import cv2


def get_patterns(path='syn', imsizes=[], crop=True):
  pattern_size = imsizes[0]
  if path == 'syn':
    np.random.seed(42)
    pattern = np.random.uniform(0,1, size=pattern_size)
    pattern = (pattern < 0.1).astype(np.float32)
    pattern.reshape(*imsizes[0])
  elif path.endswith('.npy'):
    pattern = np.load(path)
    pattern = pattern.astype(np.int32)
  else:
    pattern = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    pattern = pattern.astype(np.float32)
    # pattern /= 255
   
 # if pattern.ndim == 2:
 #   pattern = np.stack([pattern for idx in range(3)], axis=2)
  
  if crop and pattern.shape[0] > pattern_size[0] and pattern.shape[1] > pattern_size[1]:
    r0 = (pattern.shape[0] - pattern_size[0]) // 2
    c0 = (pattern.shape[1] - pattern_size[1]) // 2
    pattern = pattern[r0:r0+imsizes[0][0], c0:c0+imsizes[0][1]] 
    
  patterns = []
  for imsize in imsizes:
    #pat = cv2.resize(pattern, (imsize[1],imsize[0]), interpolation=cv2.INTER_LINEAR)
    pat = cv2.resize(pattern, (imsize[1],imsize[0]), interpolation=cv2.INTER_LINEAR)
    pat = pat.astype(np.uint8)
    patterns.append(pat)

  return patterns

def get_rotation_matrix(v0, v1):
  v0 = v0/np.linalg.norm(v0)
  v1 = v1/np.linalg.norm(v1)
  v = np.cross(v0,v1)
  c = np.dot(v0,v1)
  s = np.linalg.norm(v)
  I = np.eye(3)
  vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
  k = np.matrix(vXStr)
  r = I + k + k @ k * ((1 -c)/(s**2))
  return np.asarray(r.astype(np.float32))


def augment_image(img,rng,imgforref=None, refimg=None, illumination=None, pattern=None,disp=None,grad=None,max_shift=64,max_blur=1.5,max_noise=10.0,max_sp_noise=0.001,perspective=True):

    img=img.transpose((1,2,0))
    if imgforref is not None:
        imgforref = imgforref.transpose((1, 2, 0))
    if refimg is not None:
        refimg = refimg.transpose((1, 2, 0))
    if illumination is not None:
        illumination = illumination.transpose((1, 2, 0))
    # get min/max values of image
    min_val = np.min(img)
    max_val = np.max(img)
    
    # init augmented image
    img_aug = img
    
    # init disparity correction map
    imgforref_aug = imgforref
    refimg_aug = refimg
    illumination_aug = illumination
    pattern_aug = pattern
    disp_aug = disp
    grad_aug = grad

    # apply affine transformation
    if max_shift>1:
        
        # affine parameters
        rows,cols,_ = img.shape
        if perspective:
            src_pts = np.array([[0, 0], [0, rows], [cols, rows], [cols, 0]], dtype=np.float32)
            dst_pts = np.array([[0, rng.uniform(-max_shift,max_shift)], [0, rng.uniform(rows-max_shift,rows+max_shift)], [cols, rng.uniform(rows-max_shift,rows+max_shift)], [cols, rng.uniform(-max_shift,max_shift)]], dtype=np.float32)
            T = cv2.getPerspectiveTransform(src_pts, dst_pts)
            img_aug = cv2.warpPerspective(img_aug, T, (cols,rows))
            if grad is not None:
              grad_aug = cv2.warpPerspective(grad,T,(cols,rows))
            # disparity correction map
            # if pattern is not None:
            #   pattern_aug = cv2.warpPerspective(pattern,T,(cols,rows))
            if disp is not None:
              disp_aug = cv2.warpPerspective(disp,T,(cols,rows))
        else:
            shear = 0
            shift = 0
            shear_correction = 0
            if rng.uniform(0,1)<0: shear = rng.uniform(-max_shift,max_shift) # shear with 75% probability
            else:                     shift = rng.uniform(0,max_shift)          # shift with 25% probability
            if shear<0:               shear_correction = -shear

            # affine transformation
            a = rng.uniform(rows-max_shift,rows+max_shift)/float(rows)
            b = shift+shear_correction

            # warp image
            T = np.float32([[1,0,0],[0,a,b]])
            img_aug = cv2.warpAffine(img_aug,T,(cols,rows))
            if grad is not None:
              grad_aug = cv2.warpAffine(grad,T,(cols,rows))

            # disparity correction map
            col = a*np.array(range(rows))+b
            disp_delta = np.tile(col,(cols,1)).transpose()
            if imgforref is not None:
              imgforref_aug = cv2.warpAffine(imgforref,T,(cols,rows))
            if refimg is not None:
              refimg_aug = cv2.warpAffine(refimg_aug,T,(cols,rows))
            if illumination is not None:
              illumination_aug = cv2.warpAffine(illumination,T,(cols,rows))
            if pattern is not None:
              pattern_aug = cv2.warpAffine(pattern,T,(cols,rows))
            if disp is not None:
              # disp_aug = cv2.warpAffine(disp+disp_delta,T,(cols,rows))
              disp_aug = cv2.warpAffine(disp,T,(cols,rows))

    # gaussian smoothing
    if rng.uniform(0,1)<0.5:
        img_aug = cv2.GaussianBlur(img_aug,(5,5),rng.uniform(0.2,max_blur))
        if imgforref is not None:
            imgforref_aug = cv2.GaussianBlur(imgforref_aug,(5,5),rng.uniform(0.2,max_blur))
        
    # per-pixel gaussian noise
    img_aug = img_aug + rng.randn(*img_aug.shape)*rng.uniform(0.0,max_noise)/255.0
    if imgforref is not None:
        imgforref_aug = imgforref_aug + rng.randn(*imgforref_aug.shape)*rng.uniform(0.0,max_noise)/255.0

    # salt-and-pepper noise
    if rng.uniform(0,1)<0.5:
        ratio=rng.uniform(0.0,max_sp_noise)
        img_shape = img_aug.shape
        img_aug = img_aug.flatten()
        coord = rng.choice(np.size(img_aug), int(np.size(img_aug)*ratio))
        img_aug[coord] = max_val
        coord = rng.choice(np.size(img_aug), int(np.size(img_aug)*ratio))
        img_aug[coord] = min_val
        img_aug = np.reshape(img_aug, img_shape)

        if imgforref is not None:
            imgforref_aug = imgforref_aug.flatten()
            coord = rng.choice(np.size(imgforref_aug), int(np.size(imgforref_aug)*ratio))
            imgforref_aug[coord] = max_val
            coord = rng.choice(np.size(imgforref_aug), int(np.size(imgforref_aug)*ratio))
            imgforref_aug[coord] = min_val
            imgforref_aug = np.reshape(imgforref_aug, img_shape)
        
    # clip intensities back to [0,1]
    img_aug = np.maximum(img_aug,0.0)
    img_aug = np.minimum(img_aug,1.0)
    img_aug=img_aug.transpose((2,0,1))

    if imgforref is not None:
        imgforref_aug = np.maximum(imgforref_aug,0.0)
        imgforref_aug = np.minimum(imgforref_aug,1.0)
        imgforref_aug=imgforref_aug.transpose((2,0,1))
    if refimg is not None:
        refimg_aug=refimg_aug.transpose((2,0,1))
    if illumination is not None:
        illumination_aug=illumination_aug.transpose((2,0,1))
    # return image
    return img_aug, disp_aug, grad_aug, imgforref_aug, refimg_aug, illumination_aug, pattern_aug
