from utils import *

def main():
    
    args = arg_parser()
    
    device = detect_device(args.gpu)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    probabilities = predict(args.image, model, device, cat_to_name, args.top_k)
 
    for label, prob in zip(np.array(probabilities[1][0]), np.array(probabilities[0][0])):
        print(f"Picture of {cat_to_name.get(str(label),'')} predicted with a probability of {prob:.4f}")
        
    
if __name__ == '__main__': 
    main()