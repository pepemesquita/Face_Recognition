def process_boxes(box):
    xmin = box.rect.left()
    ymin = box.rect.top()
    xmax = box.rect.right()
    ymax = box.rect.bottom()
    return [int(xmin), int(ymin), int(xmax), int(ymax)]
