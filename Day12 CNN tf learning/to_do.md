### သင်တန်းသားများလေ့ကျင့်ရန် (Hands-on Practice Guide)

ဒီအပိုင်းမှာတော့ Neural Networks ရဲ့ အခြေခံ သဘောတရားတွေဖြစ်တဲ့ **Backpropagation**, **Optimization** နဲ့ **Hyperparameter Tuning** တို့ကို လက်တွေ့အသုံးချနိုင်ဖို့အတွက် စိတ်ဝင်စားစရာကောင်းတဲ့ Dataset တွေကို စုစည်းပေးထားပါတယ်။ PyTorch (Torch) နဲ့ Tensors တွေကို အသုံးပြုပြီး ဒီ Dataset တွေပေါ်မှာ Model တည်ဆောက်ကြည့်ပါ။

---

- [[Fashion 11-classes (Roboflow)]](https://universe.roboflow.com/data-set-uxram/fashion-classification): Roboflow မှ အဝတ်အထည်ပုံရိပ် dataset ဖြစ်ပြီး CNN နဲ့ object classification (ရှူးဖိနပ်၊ အင်္ကျီ စသဖြင့်) လုပ်ရန် အလွန်သင့်တော်ပါသည်။ 

- [[Plant Cassification 27-classes (Roboflow)]](https://universe.roboflow.com/rockpaperscissors-adtav/plants-classification-zcckg): အပင်ပုံရိပ် dataset ဖြစ်ပြီး အမျိုးအစား ၂၇ မျိုးကို CNN နဲ့ ခွဲခြားရန် သင့်တော်ပါသည်။ 

- [[0-9 digit Number (Roboflow)]](https://universe.roboflow.com/popular-benchmarks/mnist-cjkff): 0-9 digit ပုံရိပ် dataset ဖြစ်ပြီး CNN classification အတွက် basic baseline လေ့ကျင့်ဖို့ကောင်းပါတယ်။ 

- [[emotionemotion 4-classes (Roboflow)]](https://universe.roboflow.com/danyukezz/ai-emotion-detection-music-bot): မျက်နှာပုံရိပ်များကို emotion classes 4 မျိုးအဖြစ် ခွဲရန် သင့်တော်ပါသည်။

- [[recycling 7-classes (Roboflow)]](https://universe.roboflow.com/jeyoung-81arm/recyclingman): ပြန်လည်အသုံးချပစ္စည်းများကို class 7 မျိုး ခွဲရန် သင့်တော်ပါသည်။

- [[animal 10-classes (Roboflow)]](https://universe.roboflow.com/atm-ulxzv/animal-5box8): တိရစ္ဆာန်ပုံရိပ် dataset ဖြစ်ပြီး class 10 မျိုး ခွဲရန် သင့်တော်ပါသည်။

#### နမူနာ code

- [[ Fruit 11-classes (Roboflow) ]](https://universe.roboflow.com/fruit-zv7yv/fruit-dcjhh): အသီးပုံရိပ် dataset ဖြစ်ပြီး CNN နဲ့ class 11 မျိုး ခွဲရန် သင့်တော်ပါသည်။ 



> **Pro Tip:** Model ကို Train တဲ့အခါ Loss Function အနေနဲ့ `CrossEntropyLoss` ကို သုံးဖို့ မမေ့ပါနဲ့။ Backpropagation process ကို သေချာနားလည်ဖို့ `loss.backward()` နဲ့ `optimizer.step()` တို့ရဲ့ လုပ်ဆောင်ပုံကို မျက်ခြေမပြတ် စောင့်ကြည့်လေ့လာပါ။

---