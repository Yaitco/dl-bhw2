#import "./template.typ": *
#show: project.with(type: [Отчет]) 

= Датасет
В датасете 200 классов и ультрашакальные картинки 40 на 40, на каждый класс по 500 примеров. Также для него можно поcчитать 
`
mean: [0.569, 0.545, 0.493]
std: [0.188, 0.186, 0.191]
`
Отдадим 90% данных на train, так чтобы баланс классов сохранился при сплите. 

= baseline
Для начала обозначим базовую модель `BaseNet`.
```py
class BaseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.block1 = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32),
        )

        self.block2 = nn.Sequential(
            BasicBlock(32, 64, 2),
            BasicBlock(64, 64),
        )

        self.block3 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128),
        )

        self.block4 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        return x
```
Из аугментаций будем использовать только `RandomHorizontalFlip`. В качестве optimizer возьмем SGD с CosineAnnealingLR, где `momentum=0.9, weight_decay=0.0001, lr=0.01, eta_min=0.0001`. Обучив на 30 эпох, получаем $41%$ accuracy на валидации. От этого и будем плясать. 

= Аугментации
В этой главе мы рассмотрим две аугментации, а именно `RandomResizedCrop` и `RandomAffine`. Будем подбирать их параметры через на 30 эпохах. Подбор дал следующие результаты: 
- `RandomResizedCrop` был полностью убит
- Оптимальный `RandomAffine` имеет параметры
- Итоговый accuracy $44.1%$

*Тут картинка из optuna*

== Цвет и шум
Тут мы подберем `ColorJitter`, `RandomGrayscale` и `GaussianNoise`. Результаты:
-
-
-
- Итоговый accuracy $$, почти не изменился. Но статистики на train стали чуть хуже, значит, мы все-таки чуть поборолись с переобучением.

*Тут картинка из optuna*

== MixUp и CutMix
На этом этапе, предлагаю повысить сложность нашей модели и количество эпох до 50. Добавим четыре параметра к этой аугментации:
+ `p` -- вероятность срабатыания MixUp/CutMix
+ `p_mixup` -- вероятность, того, что будет делать именно MixUp
+ `mixup_alpha` -- сила MixUp
+ `cutmix_alpha` -- сила CutMix


= Модели
