from src.model_utils import load_network
import pdb 

if __name__=='__main__':
     network = load_network(model_name="DeepLabV3+_WideResNet38", num_classes=19,
                               ckpt_path=None, train=True)

     pdb.set_trace()