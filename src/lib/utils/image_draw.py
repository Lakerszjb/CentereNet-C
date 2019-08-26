from random import random as rand
import cv2
import numpy as np
import math
import copy


def draw_points(im, points, point_color=None):
    color = (rand() * 255, rand() * 255, rand() * 255) if point_color is None else point_color
    for i in range(points.shape[0]):
        cv2.circle(im, (int(points[i, 0]), int(points[i, 1])), 1, color=color, thickness=1)
    return im


def draw_box(im, box, box_color=None):
    color = (rand() * 255, rand() * 255, rand() * 255) if box_color is None else box_color
    cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=2)
    return im


def draw_track_id(im, track_id, pos, track_id_color=None):
    color = (rand() * 255, rand() * 255, rand() * 255) if track_id_color is None else track_id_color
    cv2.putText(im, text='%d' % track_id, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2)
    return im


def draw_kps(im, kps, skeleton=None, point_color=None, skeleton_color=None, kps_thresh=0, kps_show_num=False):
    # kps: (num_kps * 3, )
    kps = kps.reshape((-1, 3))
    for j in range(kps.shape[0]):
        x = int(kps[j, 0] + 0.5)
        y = int(kps[j, 1] + 0.5)
        v = kps[j, 2]
        if kps_thresh < v < 3:
            if point_color is None:
                color = (rand() * 255, rand() * 255, rand() * 255)
            elif isinstance(point_color[0], list):
                color = point_color[j % len(point_color)]
            else:
                color = point_color
            # cv2.circle(im, (x, y), 3, color=color, thickness=2)
            cv2.circle(im, (x, y), 2, color=color, thickness=3)
            if kps_show_num:
                # cv2.putText(im, '%.2f' % v, (x+3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
                cv2.putText(im, '%d' % j, (x+3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
    if skeleton is not None:
        for j in range(skeleton.shape[0]):
            p1 = skeleton[j, 0]
            p2 = skeleton[j, 1]
            x1 = int(kps[p1, 0] + 0.5)
            y1 = int(kps[p1, 1] + 0.5)
            x2 = int(kps[p2, 0] + 0.5)
            y2 = int(kps[p2, 1] + 0.5)
            if kps_thresh < kps[p1, 2] < 3 and kps_thresh < kps[p2, 2] < 3:
                if skeleton_color is None:
                    color = (rand() * 255, rand() * 255, rand() * 255)
                elif isinstance(skeleton_color[0], list):
                    color = skeleton_color[j % len(skeleton_color)]
                else:
                    color = skeleton_color
                cv2.line(im, (x1, y1), (x2, y2), color=color, thickness=3)

                # cx = (x1 + x2) / 2
                # cy = (y1 + y2) / 2
                # length = np.linalg.norm([x1 - x2, y1 - y2])
                # angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
                # polygon = cv2.ellipse2Poly((int(cx), int(cy)), (int(length/2), 2), int(angle), 0, 360, 1)
                # cv2.fillConvexPoly(im, polygon, color)
    return im


def get_edge_points(mask, min_dist=25):
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if len(contours) > 0:
        num_points = 0
        points = 0
        for j in range(len(contours)):
            if len(contours[j]) > num_points:
                num_points = len(contours[j])
                points = contours[j]
        points = points.reshape((-1, 2))
    else:
        points = np.zeros((0, 2), dtype=np.int32)
    if points.shape[0] > 0:
        mask_len = mask.shape[0] + mask.shape[1]
        min_num_points = 15
        if mask_len / min_num_points < min_dist:
            min_dist = mask_len / min_num_points
        new_points = []
        last_point = [points[0, 0], points[0, 1]]
        new_points.append(last_point)
        for i in range(1, points.shape[0]):
            dist = math.sqrt((points[i, 0] - last_point[0]) ** 2 + (points[i, 1] - last_point[1]) ** 2)
            if dist >= min_dist:
                last_point = [points[i, 0], points[i, 1]]
                new_points.append(last_point)
        points = np.array(new_points)
    # print len(points)
    return points


def get_edge_mask(mask, edge_size=5):
    pad = edge_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_size, edge_size))
    new_mask = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=mask.dtype)
    new_mask[pad:mask.shape[0] + pad, pad:mask.shape[1] + pad] = mask
    edge_mask = new_mask - cv2.erode(new_mask, kernel)
    return edge_mask[pad:mask.shape[0] + pad, pad:mask.shape[1] + pad]


def get_mask(edge_points, mask_height, mask_width):
    from skimage.draw import polygon
    edge_mask = np.zeros((mask_height, mask_width), dtype=np.bool)
    if len(edge_points) > 0:
        rr, cc = polygon(edge_points[:, 1], edge_points[:, 0])
        edge_mask[rr, cc] = 1
    return edge_mask


def draw_mask(im, mask, box=None, mask_color=None, mask_edge_color=None, scale=0.5, binary_thresh=0.5):
    if mask_color is None:
        mask_color = (rand() * 255, rand() * 255, rand() * 255)
    mask_color = np.array(mask_color).reshape((1, 1, 3))
    if mask_edge_color is None:
        mask_edge_color = (200, 200, 200)
    mask_edge_color = np.array(mask_edge_color).reshape((1, 1, 3))
    if mask.shape[:2] != im.shape[:2]:
        mask = cv2.resize(mask, (box[2] - box[0] + 1, box[3] - box[1] + 1), interpolation=cv2.INTER_LINEAR)
        mask = (mask >= binary_thresh).astype(np.uint8)
        roi_im = im[box[1]:box[3] + 1, box[0]:box[2] + 1, :]
        mask_ = mask[:, :, np.newaxis]
        roi_im[:] = roi_im - scale * roi_im * mask_ + scale * mask_color * mask_

        edge_mask = get_edge_mask(mask)
        edge_mask_ = edge_mask[:, :, np.newaxis]
        roi_im[:] = roi_im - roi_im * edge_mask_ + mask_edge_color * edge_mask_

        # edge_points = get_edge_points(mask)
        # mask = get_mask(edge_points, mask.shape[0], mask.shape[1])
        # roi_im[:] = draw_points(roi_im, edge_points, point_color=(0, 0, 255))
    else:
        mask = mask >= binary_thresh
        mask = mask[:, :, np.newaxis]
        im = im - scale * im * mask + scale * mask_color * mask
    return im


def draw_densepose_iuv_point(im, iuv):
    bbox = iuv[0]
    X = iuv[1]
    Y = iuv[2]
    IUV = iuv[3: 6]

    IUV_list = []
    for i in range(3):
        img = copy.copy(im)
        bbox = [int(round(_)) for _ in bbox]
        canvas = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        height = canvas.shape[0]
        width = canvas.shape[1]
        for j in range(len(X)):
            x = int(round(X[j] * width / 255))
            y = int(round(Y[j] * height / 255))
            if i == 0:
                cv2.circle(canvas, (x, y), 2, color=(int(255.0 * IUV[i][j] / max(IUV[i])),
                                                     int(255.0 * IUV[i][j] / max(IUV[i])),
                                                     int(255 - 255.0 * IUV[i][j] / max(IUV[i]))), thickness=2)
            else:
                cv2.circle(canvas, (x, y), 2, color=(int(255.0 * IUV[i][j] / max(IUV[i])),
                                                     int(255 - 255.0 * IUV[i][j] / max(IUV[i])),
                                                     0), thickness=2)
        IUV_list.append(img)
    iuv_img = np.vstack(IUV_list)
    return iuv_img


def draw_densepose_iuv(im, iuv, box, lines_num=9):
    iuv = np.transpose(iuv, (1, 2, 0))
    iuv = cv2.resize(iuv, (box[2] - box[0] + 1, box[3] - box[1] + 1), interpolation=cv2.INTER_LINEAR)
    roi_im = im[box[1]:box[3] + 1, box[0]:box[2] + 1, :]

    roi_im_resize = cv2.resize(roi_im, (2 * roi_im.shape[1], 2 * roi_im.shape[0]), interpolation=cv2.INTER_LINEAR)

    lines_num = int(round(lines_num * 1.0 * (box[2] - box[0] + 100) / 150))

    I = iuv[:, :, 0]
    for i in range(1, 25):
        if len(I[I == i]) == 0:
            continue

        u = np.zeros_like(I)
        v = np.zeros_like(I)
        u[I == i] = iuv[:, :, 1][I == i]
        v[I == i] = iuv[:, :, 2][I == i]

        u_num = iuv[:, :, 1][I == i]
        v_num = iuv[:, :, 2][I == i]

        for ind in range(1, lines_num):
            thred = 1.0 * ind / lines_num

            _, thresh = cv2.threshold(u, u_num.min() + thred * (u_num.max() - u_num.min()), 255, 0)
            dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
            dist_transform = np.uint8(dist_transform)

            _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = [(col * 2) for col in contours]
            cv2.drawContours(roi_im_resize, contours, -1, (int(30 + (1 - thred) * 225),
                                                           int(90 + thred * thred * (255 - 90)),
                                                           int(thred * thred * 250)), 2)

            _, thresh = cv2.threshold(v, v_num.min() + thred * (v_num.max() - v_num.min()), 255, 0)
            dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
            dist_transform = np.uint8(dist_transform)

            _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = [(col * 2) for col in contours]
            cv2.drawContours(roi_im_resize, contours, -1, (int(30 + (1 - thred) * 225),
                                                           int(90 + thred * thred * (255 - 90)),
                                                           int(thred * thred * 250)), 2)


    _, thresh = cv2.threshold(I, 0.5, 255, 0)
    dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
    dist_transform = np.uint8(dist_transform)
    _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = [(col * 2) for col in contours]
    cv2.drawContours(roi_im_resize, contours, -1, (70, 150, 0), 6)

    roi_im[:] = cv2.resize(roi_im_resize, (roi_im.shape[1], roi_im.shape[0]), interpolation=cv2.INTER_LINEAR)[:]

    return im


def attach_color_to_seg(seg, seg_color=None):
    if seg.ndim == 3:
        assert seg.shape[2] == 3
        return seg
    seg_im = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    min_ind = int(seg.min())
    max_ind = int(seg.max())
    for i in range(min_ind, max_ind + 1):
        if i <= 0:
            continue
        if seg_color is None:
            color = (rand() * 255, rand() * 255, rand() * 255)
        elif isinstance(seg_color, list):
            color = seg_color[i % len(seg_color)]
        else:
            color = seg_color
        seg_im[seg == i, 0] = color[0]
        seg_im[seg == i, 1] = color[1]
        seg_im[seg == i, 2] = color[2]
    return seg_im


def draw_seg(im, seg, box=None, seg_color=None, scale=0.5):
    seg_im = attach_color_to_seg(seg.astype(np.uint8), seg_color=seg_color)
    if im.shape[:2] == seg_im.shape[:2]:
        roi_im = im
    else:
        roi_im = im[box[1]:box[3] + 1, box[0]:box[2] + 1, :]
        if roi_im.shape[:2] != seg_im.shape[:2]:
            seg_im = cv2.resize(seg_im, (roi_im.shape[1], roi_im.shape[0]), interpolation=cv2.INTER_NEAREST)
            seg = cv2.resize(seg, (roi_im.shape[1], roi_im.shape[0]), interpolation=cv2.INTER_NEAREST)
    if seg.ndim == 3:
        seg = np.sum(seg, axis=2)
    mask = seg > 0
    mask = mask[:, :, np.newaxis]
    roi_im[:] = roi_im - scale * roi_im * mask + scale * seg_im * mask
    return im


def draw_all(im, all_boxes=None, all_classes=None, box_color=None, do_draw_box=True,
             all_segs=None, all_seg_boxes=None, seg_color=None,
             all_masks=None, all_mask_boxes=None, mask_color=None,
             all_kps=None, skeleton=None, point_color=None, skeleton_color=None, kps_thresh=0, kps_show_num=False,
             all_densepose_iuv=None, all_densepose_iuv_point=None,
             all_track_ids=None, track_id_color=None):
    # im: (h, w, 3)
    # all_boxes: (num_boxes, 4)
    # all_kps: (num_boxes, num_kps*3)
    # all_masks: (num_boxes, h, w)
    # all_segs: (num_boxes, h, w)
    from utils.colormap import colormap
    color_list = colormap()
    im_draw = copy.deepcopy(im)
    # draw boxes
    if all_boxes is not None and do_draw_box:
        all_boxes = np.round(all_boxes).astype(int)
        for i in range(len(all_boxes)):
            if all_classes is not None:
                color_idx = int(abs(all_classes[i])) - 1
            else:
                color_idx = i
            if box_color is None:
                box_color_i = color_list[color_idx % len(color_list)]
            elif isinstance(box_color, list):
                box_color_i = box_color[color_idx % len(box_color)]
            else:
                box_color_i = box_color
            im_draw = draw_box(im_draw, all_boxes[i], box_color=box_color_i)

    # draw segs
    if all_segs is not None:
        for i in range(len(all_segs)):
            if seg_color is None:
                seg_color_i = color_list
            else:
                seg_color_i = seg_color
            seg_box_i = all_seg_boxes[i].astype(int) if all_seg_boxes is not None else None
            im_draw = draw_seg(im_draw, all_segs[i], seg_box_i, seg_color=seg_color_i)

    # draw masks
    if all_masks is not None:
        for i in range(len(all_masks)):
            if mask_color is None:
                mask_color_i = color_list[i % len(color_list)]
            elif isinstance(mask_color, list):
                mask_color_i = mask_color[i % len(mask_color)]
            else:
                mask_color_i = mask_color
            mask_box_i = all_mask_boxes[i].astype(int) if all_mask_boxes is not None else None
            im_draw = draw_mask(im_draw, all_masks[i], mask_box_i, mask_color=mask_color_i)

    # draw kps
    if all_kps is not None:
        for i in range(len(all_kps)):
            if point_color is None:
                point_color_i = color_list[i % len(color_list)]
            elif isinstance(point_color, list):
                point_color_i = point_color[i % len(point_color)]
            else:
                point_color_i = point_color
            if skeleton_color is None:
                skeleton_color_i = color_list
            else:
                skeleton_color_i = skeleton_color
            im_draw = draw_kps(im_draw, all_kps[i], skeleton=skeleton,
                               point_color=point_color_i, skeleton_color=skeleton_color_i,
                               kps_thresh=kps_thresh, kps_show_num=kps_show_num)

    # draw densepose IUV
    if all_densepose_iuv_point is not None:
        if len(all_densepose_iuv_point[1]) > 0:
            im_draw = draw_densepose_iuv_point(im_draw, all_densepose_iuv_point)
    if all_densepose_iuv is not None:
        for i in range(len(all_densepose_iuv)):
            seg_densepose_box_i = all_seg_boxes[i].astype(int) if all_seg_boxes is not None else None
            im_draw = draw_densepose_iuv(im_draw, all_densepose_iuv[i], seg_densepose_box_i)

    # draw track ids
    if all_track_ids is not None:
        assert all_boxes is not None
        for i in range(len(all_track_ids)):
            if track_id_color is None:
                track_id_color_i = color_list[i % len(color_list)]
            elif isinstance(track_id_color, list):
                track_id_color_i = track_id_color[i]
            else:
                track_id_color_i = track_id_color
            x = all_boxes[i, 0] + (all_boxes[i, 2] - all_boxes[i, 0]) / 3
            im_draw = draw_track_id(im_draw, all_track_ids[i], pos=(x, all_boxes[i, 1]), track_id_color=track_id_color_i)
    return im_draw



