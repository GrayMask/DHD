import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
from pathlib import Path
import multiprocessing
import time
import json
import cv2
import os
import collections
import sys
sys.path.append('../')
import renderer
import co
from commons import get_patterns,get_rotation_matrix

def get_objs(shapenet_dir, obj_classes, num_perclass=100):

  shapenet = {'chair':      '03001627',
              'airplane':   '02691156',
              'car':        '02958343',
              'watercraft': '04530566',
              'camera':     '02942699'}

  obj_paths = []
  for cls in obj_classes:
      if cls not in shapenet.keys():
          raise Exception('unknown class name')
      ids = shapenet[cls]
      obj_path = sorted(Path(f'{shapenet_dir}/{ids}').glob('**/models/*.obj'))
      obj_paths += obj_path[:num_perclass]
  print(f'found {len(obj_paths)} object paths')

  objs = []
  for obj_path in obj_paths:
    if (obj_path.parent / "extractedn.npy").exists():
      print(f'load {obj_path.parent / "extractedv.npy"}')
      v = np.load(obj_path.parent / "extractedv.npy")
      f = np.load(obj_path.parent / "extractedf.npy")
      c = np.load(obj_path.parent / "extractedc.npy")
      n = np.load(obj_path.parent / "extractedn.npy")
    else:
      print(f'load {obj_path}')
      v, f, c, n = co.io3d.read_obj(obj_path)
      diffs = v.max(axis=0) - v.min(axis=0)
      v /= (0.5 * diffs.max())
      v -= (v.min(axis=0) + 1)
      f = f.astype(np.int32)
      np.save(obj_path.parent / "extractedv.npy",v)
      np.save(obj_path.parent / "extractedf.npy", f)
      np.save(obj_path.parent / "extractedc.npy", c)
      np.save(obj_path.parent / "extractedn.npy", n)
    obj = (v,f,c, n)
    objs.append(obj)
  print(f'loaded {len(objs)} objects')

  return objs

def get_mesh2(rng, rng_clr, ref_len, min_z=0):
  # set up background board
  verts, faces, normals, colors = [], [], [], []
  v, f, n = co.geometry.xyplane(z=0, interleaved=True)
  v[:,2] += -v[:,2].min() + rng.uniform(2,7)
  v[:,:2] *= 5e2
  v[:,2] = np.mean(v[:,2]) + (v[:,2] - np.mean(v[:,2])) * 5e2
  c = np.full(v.shape[0], rng_clr.randint(0, ref_len))
  verts.append(v)
  faces.append(f)
  normals.append(n)
  colors.append(c)

  # randomly sample 4 foreground objects for each scene
  for shape_idx in range(4):
    v, f, c, n = objs[rng.randint(0,len(objs))]
    v, f, c, n = v.copy(), f.copy(), c.copy(), n.copy()

    s = rng.uniform(0.25, 1)
    v *= s
    R = co.geometry.rotm_from_quat(co.geometry.quat_random(rng=rng))
    v = v @ R.T
    n = n @ R.T
    v[:,2] += -v[:,2].min() + min_z + rng.uniform(0.5, 3)
    v[:,:2] += rng.uniform(-0.5, 0.5, size=(1,2))

    c = np.full(v.shape[0], rng_clr.randint(0, ref_len))

    verts.append(v.astype(np.float32))
    faces.append(f)
    normals.append(n)
    colors.append(c)

  verts, faces = co.geometry.stack_mesh(verts, faces)
  normals = np.vstack(normals).astype(np.float32)
  colors = np.vstack(colors).astype(np.float32)
  return verts, faces, colors, normals

def get_mesh(rng, rng_clr, ref_len, x, y, min_z=0):
  # set up background board
  verts, faces, normals, colors = [], [], [], []
  v, f, n = co.geometry.xyplane(z=0, interleaved=True)
  v[:,2] += -v[:,2].min() + rng.uniform(0.53,0.58)
  v[:,:2] *= 5e2
  v[:,2] = np.mean(v[:,2]) + (v[:,2] - np.mean(v[:,2])) * 5e2
  c = np.full(v.shape[0], rng_clr.randint(0, ref_len))
  verts.append(v)
  faces.append(f)
  normals.append(n)
  colors.append(c)

  # randomly sample 4 foreground objects for each scene
  for shape_idx in range(4):
    objidx = rng.randint(0,len(objs))
    v, f, c, n = objs[objidx]
    v, f, c, n = v.copy(), f.copy(), c.copy(), n.copy()

    s = rng.uniform(0.3, 0.33)
    v *= s
    R = co.geometry.rotm_from_quat(co.geometry.quat_random(rng=rng))
    v = v @ R.T
    n = n @ R.T
    v[:,2] += -v[:,2].min() + min_z + rng.uniform(0.40, 0.45)
    v[:,0] += v[:,2]*x+rng.uniform(-0.15, 0.15)
    v[:,1] += v[:,2]*y+rng.uniform(-0.15, 0.15)

    c_idx=rng_clr.randint(ref_len, size=(c.max()+1))#np.full(v.shape[0], )

    c = c_idx[c]

    verts.append(v.astype(np.float32))
    faces.append(f)
    normals.append(n)
    colors.append(c)

  verts, faces = co.geometry.stack_mesh(verts, faces)
  normals = np.vstack(normals).astype(np.float32)
  colors = np.hstack(colors).astype(np.int32)
  return verts, faces, colors, normals


def create_data(out_root, idx, n_samples, imsize, patterns, reflectance, camerasensitivity, illumination, wavelength, K, shiftcamera, shiftpattern, baseline, blend_im, noise, maxdisp, mindisp, track_length=4):

  tic = time.time()
  rng = np.random.RandomState()
  rng_clr = np.random.RandomState()

  rng.seed(idx)
  rng_clr.seed(idx)

  x_center=(imsize[1]/2-K[0,2]+shiftcamera)/K[0,0]
  y_center=(imsize[0]/2-K[1,2])/K[1,1]

  verts, faces, colors, normals = get_mesh(rng, rng_clr, reflectance.shape[0],x_center,y_center)
  data = renderer.PyRenderInput(verts=verts.copy(), colors=colors.copy(), normals=normals.copy(), faces=faces.copy())
  print(f'loading mesh for sample {idx+1}/{n_samples} took {time.time()-tic}[s]')


  # let the camera point to the center
  center = np.array([0,0,0.4], dtype=np.float32)

  basevec =  np.array([-baseline,0,0], dtype=np.float32)
  unit = np.array([0,0,1],dtype=np.float32)

  cam_x_ = rng.uniform(-0.05,0.05)
  cam_y_ = rng.uniform(-0.05,0.05)
  cam_z_ = rng.uniform(0,0)

  ret = collections.defaultdict(list)
  blend_im_rnd = np.clip(blend_im + rng.uniform(-0.1,0.1), 0,1)

  # capture the same static scene from different view points as a track
  for ind in range(track_length):

    cam_x = cam_x_
    cam_y = cam_y_
    cam_z = cam_z_
    
    tcam = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    if np.linalg.norm(tcam[0:2])<1e-9:
      Rcam = np.eye(3, dtype=np.float32)
    else:
      Rcam = get_rotation_matrix(center, center-tcam)

    tproj = tcam + basevec 
    Rproj = Rcam

    ret['R'].append(Rcam)
    ret['t'].append(tcam)

    cams = []
    projs = []

    # render the scene at multiple scales
    scales = [1, 0.5, 0.25, 0.125]

    for scale in scales:
      fx = K[0,0] * scale
      fy = K[1,1] * scale
      pxcam = (K[0,2]-shiftcamera) * scale
      pxpro = (K[0,2]-shiftpattern) * scale
      py = K[1,2] * scale
      im_height = imsize[0] * scale
      im_width = imsize[1] * scale
      cams.append( renderer.PyCamera(fx,fy,pxcam,py, Rcam, tcam, im_width, im_height) )
      projs.append( renderer.PyCamera(fx,fy,pxpro,py, Rproj, tproj, im_width, im_height) )


    for s, cam, proj, pattern in zip(itertools.count(), cams, projs, patterns):
      fl = K[0,0] / (2**s)
      shiftcameras=shiftcamera/ (2**s)
      shiftpatterns=shiftpattern/ (2**s)
      shader = renderer.PyShader(0,1.5,0.0,10)
      pyrenderer = renderer.PyRenderer(cam, shader, wavelength=wavelength, engine='gpu')
      pyrenderer.mesh_proj(data, proj, pattern, reflectance, camerasensitivity, illumination, d_alpha=0, d_beta=0.35)

      # get the reflected laser pattern $R$
      im = pyrenderer.color().copy()
      refimg = pyrenderer.reflectance().copy()
      depth = pyrenderer.depth().copy()
      disp = baseline * fl / depth - shiftcameras + shiftpatterns
      mask = depth > 0

      # get the ambient image $A$
      ambient = pyrenderer.normal().copy()

      # get the noise free IR image $J$ 
      if s==0:
        ret[f'ambient{s}'].append( ambient[None].astype(np.float32) )

      ret[f'im{s}'].append( im[None].astype(np.float32))
      ret[f'refimg{s}'].append( refimg[None].astype(np.float32))
      dmax=disp.max()
      dmin=disp.min()
      maxdisp = max(maxdisp,dmax)
      mindisp = min(mindisp,dmin)
      ret[f'disp{s}'].append(disp[None].astype(np.float32))
      print(f'Disp Min: {dmin}; Disp Max: {dmax}. Whole Min: {mindisp}; Whole Max: {maxdisp}')

  for key in ret.keys():
    ret[key] = np.stack(ret[key], axis=0)

  # save to files
  out_dir = out_root / f'{idx:08d}'
  out_dir.mkdir(exist_ok=True, parents=True)
  for k,val in ret.items():
    for tidx in range(track_length):
      v = val[tidx]
      out_path = out_dir / f'{k}_{tidx}.npy'
      np.save(out_path, v)
  np.save( str(out_dir /'blend_im.npy'), blend_im_rnd)

  print(f'create sample {idx+1}/{n_samples} took {time.time()-tic}[s]')

  return maxdisp, mindisp



if __name__=='__main__':

  np.random.seed(42)
  
  # output directory
  with open('../para/config.json') as fp:
   config = json.load(fp)
   data_root = Path(config['DATA_ROOT'])
   shapenet_root = config['SHAPENET_ROOT']
  
  data_type = 'syn'
  out_root = data_root / f'{data_type}'
  out_root.mkdir(parents=True, exist_ok=True)

  # load shapenet models 
  # obj_classes = ['chair','car']
  obj_classes = ['airplane','watercraft','camera']
  objs = get_objs(shapenet_root, obj_classes)
  
  # camera parameters
  imsize = (480, 640)
  imsizes = [(imsize[0]//(2**s), imsize[1]//(2**s)) for s in range(4)]
  with open(str('../para/campara.pkl'), 'rb') as f:
      campara = pickle.load(f)
  K = campara['K']
  baseline = campara['baseline']
  shiftcamera= campara['shiftcamera']
  shiftpattern= campara['shiftpattern']
  print(K)

  focal_lengths = [K[0,0]/(2**s) for s in range(4)]
  blend_im = 0.6
  noise = 0
  
  # capture the same static scene from different view points as a track
  track_length = 1#4
  
  # load pattern image
  pattern_path = '../para/pattern_croped.png'
  pattern_crop = True
  patterns = get_patterns(pattern_path, imsizes, pattern_crop)

  # load reflectance
  reflectance = np.loadtxt('../para/reflectance.txt', dtype=np.float32, delimiter=',')
  wavelength = reflectance.shape[1]

  # load camera sensitivity
  camerasensitivity = np.loadtxt('../para/camerasensitivity.txt', dtype=np.float32, delimiter=',')

  # load illumination
  illumination = np.loadtxt('../para/illumination.txt', dtype=np.float32, delimiter=',')

  patterns_illu=[]
  for index, value in enumerate(patterns):
    value=value.astype(np.float16)/255
    ill_pat = value@illumination[:3,:]
    print(ill_pat.shape)
    patterns_illu.append(ill_pat)
  settings = {
   'imsizes': imsizes,
   'patterns_3ch': patterns,
   'patterns_illu':patterns_illu,
   'camerasensitivity':camerasensitivity,
   'focal_lengths': focal_lengths,
   'baseline': baseline,
   'K': K,
   'shiftcamera': shiftcamera,
    'shiftpattern': shiftpattern,
  }
  out_path = out_root / f'settings.pkl'
  print(f'write settings to {out_path}')
  with open(str(out_path), 'wb') as f:
   pickle.dump(settings, f, pickle.HIGHEST_PROTOCOL)

  maxdisp=0
  mindisp=sys.float_info.max

  # start the job
  n_samples = 2**10 + 2**13
  #n_samples = 2**8# + 2**13
  for idx in range(n_samples):
    args = (out_root, idx, n_samples, imsize, patterns, reflectance, camerasensitivity, illumination, wavelength, K, shiftcamera, shiftpattern, baseline, blend_im, noise, maxdisp, mindisp, track_length)
    maxdisp, mindisp = create_data(*args)
