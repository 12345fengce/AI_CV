# -*-coding:utf-8-*-
img_size = (416, 416)

prior_boxes = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

prior_areas = {
    13: [x*y for x, y in prior_boxes[13]],
    26: [x*y for x, y in prior_boxes[26]],
    52: [x*y for x, y in prior_boxes[52]]
}
