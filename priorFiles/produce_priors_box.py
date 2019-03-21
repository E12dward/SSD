"""生成各featuremap层的default box信息"""


from keras.engine.topology import Layer
import numpy as np
import pickle

class PriorBox(Layer):
    """Generate the prior boxes of designated sizes and aspect ratios.

    # Arguments
        img_size: Size of the input image as tuple (w, h).
        min_size: Minimum box size in pixels.
        max_size: Maximum box size in pixels.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
    """
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1,0.1,0.2,0.2], clip=True, layer_shape=(8,8),**kwargs):
        self.layer_shape=layer_shape
        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        super(PriorBox, self).__init__(**kwargs)



    def compute_default_box(self):
        layer_width = self.layer_shape[0]
        layer_height = self.layer_shape[1]
        img_width = self.img_size[0]
        img_height = self.img_size[1]
        # define prior boxes shapes
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        # define centers of prior boxes
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(self.aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        # define variances
        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        
        return prior_boxes
    
    
    
img_size=(300,300)
default_box_layer1=PriorBox(img_size,30,[],aspect_ratios=[2],layer_shape=(38,38)).compute_default_box()
default_box_layer2=PriorBox(img_size,60,114,aspect_ratios=[2,3],layer_shape=(19,19)).compute_default_box()
default_box_layer3=PriorBox(img_size,114,168,aspect_ratios=[2,3],layer_shape=(10,10)).compute_default_box()
default_box_layer4=PriorBox(img_size,168,222,aspect_ratios=[2,3],layer_shape=(5,5)).compute_default_box()
default_box_layer5=PriorBox(img_size,222,276,aspect_ratios=[2,3],layer_shape=(3,3)).compute_default_box()
default_box_layer6=PriorBox(img_size,276,330,aspect_ratios=[2,3],layer_shape=(1,1)).compute_default_box()

default_box=np.concatenate((default_box_layer1,default_box_layer2,default_box_layer3,default_box_layer4,
                            default_box_layer5,default_box_layer6),axis=0) 
pickle.dump(default_box,open("default_box_information.pkl","wb"))  

'''
priors1 = pickle.load(open('prior_boxes_ssd300.pkl', 'rb')) 
priors2 = pickle.load(open('default_box_information.pkl', 'rb')) 
print(priors1[:5,:5])
print(priors2[:5,:5])
'''




