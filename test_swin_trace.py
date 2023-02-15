import timm
import torch
if __name__ == "__main__":    
    model = timm.create_model("swin_s3_tiny_224", pretrained=False)
    model.eval()
    input = torch.randn(1, 3, 224, 224)
    
    tracemodel = torch.jit.trace(model,input)

    x= torch.randn(5, 3, 224, 224)
    y = model(x)
    y_traced = tracemodel(x)
    print("diff between trace and untraced:", torch.max(abs(y-y_traced)))