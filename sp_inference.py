#!/usr/bin/env python


import argparse
import glob
import numpy as np
import os
import time
import torch.nn.functional as F
import cv2
import torch
import sys
import torch.nn as nn
from tools import*
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')



# Define the SuperPointNet_pytorch model
class SuperPointNet_pytorch(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self):
        super(SuperPointNet_pytorch, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        self.relu = nn.ReLU(inplace=True)
        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))

        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        # Normalize the descriptors.
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        output = {'semi': semi, 'desc': desc}

        return semi, desc

class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh, cuda=False):
    self.name = 'SuperPoint'
    self.cuda = cuda
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
    self.cell = 8  # Size of each output cell. Keep this fixed.
    self.border_remove = 4  # Remove points this close to the border.

    # Load the network in inference mode.
    self.net = SuperPointNet_pytorch()

    # Load the weights (whether it's a full checkpoint or just model weights)
    checkpoint = torch.load(weights_path, map_location='cuda' if cuda else 'cpu')
    if 'model_state_dict' in checkpoint:
      # If the weights file is a checkpoint, extract the model state dict
      state_dict = checkpoint['model_state_dict']
    else:
      # If it's just the model weights, load the full state dict directly
      state_dict = checkpoint

    # Load the state dictionary into the network
    self.net.load_state_dict(state_dict)

    if cuda:
      self.net = self.net.cuda()

    self.net.eval()

  def nms_fast(self, in_corners, H, W, dist_thresh):

    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    """
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, desc, heatmap

class PointTracker:
    """ Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking, using OpenCV's BFMatcher.
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl+2))
        self.track_count = 0
        self.max_score = 9999
        # OpenCV BFMatcher with L2 norm (default for SuperPoint descriptors).
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def update(self, pts, desc):
        """ Add a new set of point and descriptor observations to the tracker. """

        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        assert pts.shape[1] == desc.shape[1]

        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))

        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)

        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)

        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()

        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))

        # Try to append to existing tracks using BFMatcher.
        matched = np.zeros((pts.shape[1])).astype(bool)
        if self.last_desc.shape[1] > 0:
            matches = self.matcher.match(desc.T.astype(np.float32), self.last_desc.T.astype(np.float32))
            matches = [m for m in matches if m.distance < self.nn_thresh]

            for match in matches:
                id1 = match.trainIdx + offsets[-2]
                id2 = match.queryIdx + offsets[-1]
                found = np.argwhere(self.tracks[:, -2] == id1)
                if found.shape[0] > 0:
                    matched[match.queryIdx] = True
                    row = int(found)
                    self.tracks[row, -1] = id2
                    if self.tracks[row, 1] == self.max_score:
                        self.tracks[row, 1] = match.distance
                    else:
                        track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                        frac = 1. / float(track_len)
                        self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match.distance

        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num

        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]

        # Store the last descriptors.
        self.last_desc = desc.copy()

    def get_offsets(self):
        """ Iterate through list of points and accumulate an offset value. """
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts)-1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def get_tracks(self, min_length):
      """ Retrieve point tracks of a given minimum length.
      """
      if min_length < 1:
        raise ValueError('\'min_length\' too small.')
      valid = np.ones((self.tracks.shape[0])).astype(bool)
      good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
      # Remove tracks which do not have an observation in most recent frame.
      not_headless = (self.tracks[:, -1] != -1)
      keepers = np.logical_and.reduce((valid, good_len, not_headless))
      returned_tracks = self.tracks[keepers, :].copy()
      return returned_tracks

    def draw_tracks(self, out, tracks):
      """
      Visualize tracks all overlayed on a single image with green color.
      """
      pts_mem = self.all_pts
      N = len(pts_mem)  # Number of frames in memory.
      offsets = self.get_offsets()
      stroke = 1  # Line thickness.
      radius = 2  # Marker size.

      # Define green color.
      green_color = (0, 255, 0)

      for track in tracks:
          for i in range(N - 1):
              if track[i + 2] == -1 or track[i + 3] == -1:
                  continue
              offset1 = offsets[i]
              offset2 = offsets[i + 1]
              idx1 = int(track[i + 2] - offset1)
              idx2 = int(track[i + 3] - offset2)
              pt1 = pts_mem[i][:2, idx1]
              pt2 = pts_mem[i + 1][:2, idx2]
              p1 = (int(round(pt1[0])), int(round(pt1[1])))
              p2 = (int(round(pt2[0])), int(round(pt2[1])))
              # Draw green lines.
              cv2.line(out, p1, p2, green_color, thickness=stroke, lineType=cv2.LINE_AA)
              # Draw green endpoints for the last point.
              if i == N - 2:
                  cv2.circle(out, p2, radius, green_color, -1, lineType=cv2.LINE_AA)

class VideoStreamer:
    """Class to handle video streams and image inputs."""
    
    def __init__(self, source, cam_id=0, height=480, width=640, skip=1, img_glob='*.png'):
        self.source = source
        self.height = height
        self.width = width
        self.skip = skip
        self.current_frame = 0
        self.max_frames = float('inf')  # Default to an infinite stream.
        self.frame_list = None
        self.cap = None

        if source == "camera":
            print("==> Initializing webcam input.")
            self.cap = cv2.VideoCapture(cam_id)
            if not self.cap.isOpened():
                raise IOError("Cannot access the webcam.")
        elif os.path.isfile(source):  # Assume it's a video file.
            print("==> Initializing video file input.")
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file: {source}")
            self.max_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.skip
        elif os.path.isdir(source):  # Assume it's a directory of images.
            print("==> Initializing image directory input.")
            self.frame_list = sorted(glob.glob(os.path.join(source, img_glob)))[::self.skip]
            if len(self.frame_list) == 0:
                raise IOError(f"No images found in directory {source} with glob pattern {img_glob}")
            self.max_frames = len(self.frame_list)
        else:
            raise ValueError("Invalid input source. Must be 'camera', a video file, or an image directory.")

    def _resize_and_normalize(self, img):

        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.0
        return img

    def next_frame(self):

        if self.current_frame >= self.max_frames:
            return None, False

        if self.cap:  # Handle webcam or video input.
            ret, frame = self.cap.read()
            if not ret:
                return None, False
            if self.source != "camera":  # For video files, seek specific frames.
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame * self.skip)
            frame = self._resize_and_normalize(frame)
        else:  # Handle image directory input.
            frame_path = self.frame_list[self.current_frame]
            frame = cv2.imread(frame_path)
            if frame is None:
                raise IOError(f"Error reading image: {frame_path}")
            frame = self._resize_and_normalize(frame)

        self.current_frame += 1
        return frame, True

    def reset(self):
        """Reset the streamer to its initial state."""
        self.current_frame = 0
        if self.cap and self.source != "camera":  # For video files.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        """Release resources."""
        if self.cap:
            self.cap.release()

# class double_conv(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#     def __init__(self, in_ch, out_ch):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class inconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(inconv, self).__init__()
#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch)
#         )

#     def forward(self, x):
#         x = self.mpconv(x)
#         return x


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super(up, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)

#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2))

#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x


# class outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(outconv, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 1)

#     def forward(self, x):
#         x = self.conv(x)
#         return x

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])




if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--weights_path', type=str, default='/weights/ica_weights_1.pth.tar',
                        help='Path to pretrained weights file.')
    parser.add_argument('--img_glob', type=str, default='*.png',
                        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
                        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--show_extra', action='store_true',
                        help='Show extra debug outputs (default: False).')
    parser.add_argument('--H', type=int, default=480,
                        help='Input image height (default: 480).')
    parser.add_argument('--W', type=int, default=640,
                        help='Input image width (default: 640).')
    parser.add_argument('--display_scale', type=int, default=3,
                        help='Factor to scale output visualization (default: 3).')
    parser.add_argument('--min_length', type=int, default=2,
                        help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=5,
                        help='Maximum length of point tracks (default: 5).')
    parser.add_argument('--nms_dist', type=int, default=4,
                        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.010,
                        help='Detector confidence threshold (default: 0.010).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
                        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA GPU to speed up network processing speed (default: False)')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--write', action='store_true',
                        help='Save output frames to a directory (default: False)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
                        help='Directory where to write output frames (default: tracker_outputs/).')
    opt = parser.parse_args()
    print(opt)

    # Initialize VideoStreamer to load input images.
    vs = VideoStreamer(opt.input, cam_id=opt.camid, height=opt.H, width=opt.W, skip=opt.skip, img_glob=opt.img_glob)

    # Load the SuperPoint network.
    fe = SuperPointFrontend(weights_path=opt.weights_path,
                            nms_dist=opt.nms_dist,
                            conf_thresh=opt.conf_thresh,
                            nn_thresh=opt.nn_thresh,
                            cuda=opt.cuda)
    print('==> Network succesfully loaded.<==')

    # Initialize PointTracker with OpenCV BFMatcher logic.
    tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

    # Create visualization window if not disabled.
    if not opt.no_display:
        win = 'SuperPoint Tracker'
        cv2.namedWindow(win)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Create output directory if write option is enabled.
    if opt.write:
        print('==> Will write outputs to %s' % opt.write_dir)
        if not os.path.exists(opt.write_dir):
            os.makedirs(opt.write_dir)

    print('==> Running Demo.')


    print('==> Running Demo.')
    while True:
        start = time.time()

        # Get a new frame.
        img, status = vs.next_frame()
        if not status:
            break

        # Run SuperPoint inference to get points and descriptors.
        start1 = time.time()
        pts, desc, heatmap = fe.run(img)
        end1 = time.time()

        # Update the tracker with new points and descriptors.
        tracker.update(pts, desc)

        # Retrieve valid tracks from the tracker.
        tracks = tracker.get_tracks(opt.min_length)

        # Primary output - Show point tracks overlayed on top of input image.
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        tracks[:, 1] /= float(fe.nn_thresh)  # Normalize track scores.
        tracker.draw_tracks(out1, tracks)

        # Optionally show other outputs (heatmap, raw detections).
        if opt.show_extra:
            out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
            for pt in pts.T:
                pt1 = (int(round(pt[0])), int(round(pt[1])))
                cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            # Heatmap visualization.
            if heatmap is not None:
                min_conf = 0.001
                heatmap[heatmap < min_conf] = min_conf
                heatmap = -np.log(heatmap)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
                out3 = myjet[np.round(np.clip(heatmap * 10, 0, 9)).astype('int'), :]
                out3 = (out3 * 255).astype('uint8')
            else:
                out3 = np.zeros_like(out2)

            combined_out = np.hstack((out1, out2, out3))
            display_out = combined_out
        else:
            display_out = out1

        # Display the final visualization.
        if not opt.no_display:
            cv2.imshow(win, display_out)
            key = cv2.waitKey(opt.waitkey) & 0xFF
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break

        # Optionally save output frames to disk.
        if opt.write:
            out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.current_frame)
            print('Writing image to %s' % out_file)
            cv2.imwrite(out_file, display_out)

        end = time.time()
        net_t = (1. / float(end1 - start))
        total_t = (1. / float(end - start))
        if opt.show_extra:
            print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).' \
                  % (vs.current_frame, net_t, total_t))

    # Clean up and close windows.
    vs.close()
    cv2.destroyAllWindows()
    print('==> Finished Demo.')
