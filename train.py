from fashion_mnist import load_fashion_mnist
from model.deep_convnet import DeepConvNet
from common.trainer import Trainer
import matplotlib.pyplot as plt

# 데이터 로드
(x_train, t_train), (x_test, t_test) = load_fashion_mnist()

# 네트워크 생성
network = DeepConvNet()

# Trainer 설정
trainer = Trainer(
    network,
    x_train,
    t_train,
    x_test,
    t_test,
    epochs=30,
    mini_batch_size=100,
    optimizer='Adam',
    optimizer_param={'lr': 0.0005},
    evaluate_sample_num_per_epoch=1000
)

# 학습 시작
trainer.train()

# Epoch별 정확도 출력
print("\n========== Epoch별 Accuracy ==========")

for epoch in range(len(trainer.train_acc_list)):
    print(
        f"Epoch {epoch+1:2d} | "
        f"Train Acc: {trainer.train_acc_list[epoch]:.4f} | "
        f"Test Acc: {trainer.test_acc_list[epoch]:.4f}"
    )

# 그래프 출력
x = range(1, len(trainer.train_acc_list) + 1)

plt.figure(figsize=(8, 5))
plt.plot(x, trainer.train_acc_list, marker='o', label='Train Accuracy')
plt.plot(x, trainer.test_acc_list, marker='s', label='Test Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fashion MNIST CNN Accuracy")
plt.xticks(range(1, len(trainer.train_acc_list) + 1))
plt.grid(True)
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()

# 최종 결과
print("\n========== 최종 결과 ==========")
print(f"Final Train Accuracy : {trainer.train_acc_list[-1]:.4f}")
print(f"Final Test Accuracy  : {trainer.test_acc_list[-1]:.4f}")

print(f"Best Train Accuracy  : {max(trainer.train_acc_list):.4f}")
print(f"Best Test Accuracy   : {max(trainer.test_acc_list):.4f}")