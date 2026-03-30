from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from model import FaceNet
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
class UTKFace (Dataset):
    def __init__(self):
        super().__init__()
        self.ds = load_dataset("nu-delta/utkface", split = "train")
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            #got this shit from gemini
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index):
        item = self.ds.__getitem__(index)
        image = self.transform(item['image'].convert("RGB"))
        age = torch.tensor([item['age']], dtype=torch.float32)
        gender = torch.tensor([1 if item['gender'] == 'Male' else 0],dtype=torch.float32)

        return image, age, gender

ds = UTKFace()
dl = DataLoader(dataset=ds, batch_size=100,shuffle=True)

model = FaceNet(3,200)
model = model.to(device)
epochs = 10
optimizer = torch.optim.AdamW(params=model.parameters(),lr = 1e-4)
age_loss = torch.nn.L1Loss()
gender_loss = torch.nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    total_loss = 0
    for image, age, gender in tqdm(dl):
        image = image.to(device)
        age = age.to(device)
        gender = gender.to(device)

        p_age, p_gender = model(image)

        loss_age = age_loss(p_age,age)
        loss_gender = gender_loss(p_gender, gender)
        loss = loss_age + loss_gender
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    torch.save(model.state_dict(), "model_weights.pt")
    print(f"Epoch: {epoch}, Loss:{total_loss/len(dl)}")