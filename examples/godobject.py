import os
import numpy
import cv2 as cv
import math


#   Путь к заданию п.1
pathAllababah = r'C:\Users\Evgeny\Desktop\PycharmProjects\Lab_3\Lab_3_task\img_zadan\allababah'
#   Путь к заданию п.2
pathTeplovizor = r'C:\Users\Evgeny\Desktop\PycharmProjects\Lab_3\Lab_3_task\img_zadan\teplovizor'
#   Путь к заданию п.3
pathRoboti = r'C:\Users\Evgeny\Desktop\PycharmProjects\Lab_3\Lab_3_task\img_zadan\roboti'
#   Путь к заданию п.4
pathGk = r'C:\Users\Evgeny\Desktop\PycharmProjects\Lab_3\Lab_3_task\img_zadan\gk'


class Aims(object):
    def __init__(self, image, hMin = 0, sMin = 0, vMin = 0, hMax = 179, sMax = 255, vMax = 255):
        """Constructor"""
        self.image = image
        #   Меняем цветовую модель с BGR на HSV
        self.hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        self.lowerThreshold = numpy.array([hMin, sMin, vMin])
        self.upperThreshold = numpy.array([hMax, sMax, vMax])
        self.lightCG = numpy.zeros(2)
        self.cgX = 0
        self.cgY = 0
        self.redOnCenter = numpy.zeros(3, dtype=numpy.int32)
        self.greenOnCenter = numpy.zeros(3,  dtype=numpy.int32)
        self.blueOnCenter = numpy.zeros(3,  dtype=numpy.int32)
        self.closestToLight = numpy.zeros(3,  dtype=numpy.int32)

        self.dist = 0
        self.contourSize = numpy.zeros(2, dtype=numpy.uint8)   # допустимый диапазон найденного контура

        self.blueThresh = numpy.zeros(2, dtype=numpy.uint8)
        self.greenThresh = numpy.zeros(2, dtype=numpy.uint8)
        self.redThresh = numpy.zeros(2, dtype=numpy.uint8)


    def writeThreshold(self, hMin = 0, sMin = 0, vMin = 0, hMax = 179, sMax = 255, vMax = 255):
        self.lowerThreshold[0] = hMin
        self.lowerThreshold[1] = sMin
        self.lowerThreshold[2] = vMin
        self.upperThreshold[0] = hMax
        self.upperThreshold[1] = sMax
        self.upperThreshold[2] = vMax

    def changeHmin(self, dataTrackBar):
        # получение текущего значения ползунка
        self.lowerThreshold[0] = dataTrackBar

    def changeSmin(self, dataTrackBar):
        # получение текущего значения ползунка
        self.lowerThreshold[1] = dataTrackBar

    def changeVmin(self, dataTrackBar):
        # получение текущего значения ползунка
        self.lowerThreshold[2] = dataTrackBar

    def changeHmax(self, dataTrackBar):
        # получение текущего значения ползунка
        self.upperThreshold[0] = dataTrackBar

    def changeSmax(self, dataTrackBar):
        # получение текущего значения ползунка
        self.upperThreshold[1] = dataTrackBar

    def changeVmax(self, dataTrackBar):
        # получение текущего значения ползунка
        self.upperThreshold[2] = dataTrackBar

    def changePointMin(self, dataTrackBar):
        # получение текущего значения ползунка
        self.contourSize[0] = dataTrackBar

    def changePointMax(self, dataTrackBar):
        # получение текущего значения ползунка
        self.contourSize[1] = dataTrackBar

    def eroseAndDilate(self, image=None):
        if image is None:
            hsvFiltered = self.hsvFilter()
        else:
            hsvFiltered = image # Если необ выполнить эрозию своего изображения, оно д.б. Бинарным(черно/белым)!!!
        while True:
            cv.imshow('filtered', hsvFiltered)
            k = cv.waitKey(0)
            if chr(k) == 'e':
                hsvFiltered = cv.erode(hsvFiltered, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)), iterations=1)
            elif chr(k) == 'd':
                hsvFiltered = cv.dilate(hsvFiltered, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)), iterations=1)
            elif chr(k) == 'f':
                hsvFiltered = cv.erode(hsvFiltered, cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30)), iterations=1)
            elif chr(k) == 's':
                cv.imwrite('light.jpg', hsvFiltered)
            elif k == 27:
                break
        cv.destroyWindow('filtered')
        cv.destroyWindow('original')
        return hsvFiltered

    def writeThresholdBar(self):
        img = self.image.copy()
        cv.imshow('original', img)
        # сначала необ создать окошко, в котором будут наши ползунки
        cv.namedWindow('Regulation of filter parameters')
        cv.namedWindow('image')
        # создаем ползунки - первый параметр - имя ползунка, второй - имя окошка, в котором будет ползунок,
        # третий/четвертый: мин/макс значение ползунка,
        # пятый - функция, которая будет вызываться при изменении положения ползунка

        cv.createTrackbar('Hmin', 'Regulation of filter parameters', 0, 179, self.changeHmin)
        cv.createTrackbar('Smin', 'Regulation of filter parameters', 0, 255, self.changeSmin)
        cv.createTrackbar('Vmin', 'Regulation of filter parameters', 0, 255, self.changeVmin)

        cv.createTrackbar('Hmax', 'Regulation of filter parameters', 0, 179, self.changeHmax)
        cv.createTrackbar('Smax', 'Regulation of filter parameters', 0, 255, self.changeSmax)
        cv.createTrackbar('Vmax', 'Regulation of filter parameters', 0, 255, self.changeVmax)

        hsvTemp = None   # Начальная инициализация
        while True:   # здесь подбираем пороговые значения
            k = cv.waitKey(1) & 0xFF    # если нажата клавиша Esc - пороговые значения подобраны - переход к след шагу
            if k == 27:
                break
            #   Применяем цветовой фильтр
            hsvTemp = self.hsvFilter()
            cv.imshow('image', hsvTemp)
        cv.destroyWindow('original')
        cv.destroyWindow('image')
        cv.destroyWindow('Regulation of filter parameters')

    def writeContourSizeBar(self):
        #   Поиск контуров и сохранение их в переменную contours
        cv.namedWindow('regContours')

        cv.namedWindow('Regulation of contour size filter')
        cv.createTrackbar('Pmin', 'Regulation of contour size filter', 0, 50, self.changePointMin)
        cv.createTrackbar('Pmax', 'Regulation of contour size filter', 0, 200, self.changePointMax)
        contours = self.findContours()
        while True:
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            exmp = self.image.copy()
            self.blueOnCenter[0] = 5000
            self.greenOnCenter[0] = 5000
            self.redOnCenter[0] = 5000
            for number in range(len(contours)):
                #   Фильтрация контуров-шумов
                if contours[number].shape[0] >= self.contourSize[0] and contours[number].shape[0] < self.contourSize[1]:
                    #   Нахождение момента контура
                    middle = 0  # нахлождение усредненного значения цвета крышки робота
                    for pixel in contours[number]:
                        middle += self.hsvImage[pixel[0][1]][pixel[0][0]][0]
                    middle /= contours[number].shape[0]

                    moments = cv.moments(contours[number])
                    #   Нахождение центра масс робота
                    self.cgX = int(moments['m10'] / moments['m00'])
                    self.cgY = int(moments['m01'] / moments['m00'])

                    # вычисление расстояния между центрами масс лампы и робота
                    dist = math.sqrt(pow((self.cgX - self.lightCG[0]), 2) + pow((self.cgY - self.lightCG[1]), 2))

                    if dist < 50:   # фильтрация контура абажура лампы
                        continue

                    if self.hsvImage[self.cgY][self.cgX][0] > 81 and self.hsvImage[self.cgY][self.cgX][0] < 115:
                        color = (255, 0, 0) # фильтр синей команды
                        if dist < self.blueOnCenter[0]:
                            self.blueOnCenter[0] = dist
                            self.blueOnCenter[1:] = self.cgX, self.cgY
                    elif self.hsvImage[self.cgY][self.cgX][0] > 63 and self.hsvImage[self.cgY][self.cgX][0] < 80:
                        color = (0, 255, 0) # фильтр зеленой команды
                        if dist < self.greenOnCenter[0]:
                            self.greenOnCenter[0] = dist
                            self.greenOnCenter[1:] = self.cgX, self.cgY
                    else:
                        color = (0, 0, 255) # фильтр красной команды
                        if dist < self.redOnCenter[0]:
                            self.redOnCenter[0] = dist
                            self.redOnCenter[1:] = self.cgX, self.cgY
                    # обводка крышки робота цветом его команды
                    cv.drawContours(image=exmp, contours=contours, contourIdx=number, color=color, thickness=2)
            cv.circle(exmp, (self.blueOnCenter[1], self.blueOnCenter[2]), 3, (255, 0, 0), -1)
            cv.circle(exmp, (self.greenOnCenter[1], self.greenOnCenter[2]), 3, (0, 255, 0), -1)
            cv.circle(exmp, (self.redOnCenter[1], self.redOnCenter[2]), 3, (0, 0, 255), -1)
            cv.imshow('regContours', exmp)
        cv.destroyWindow('regContours')
        cv.destroyWindow('Regulation of contour size filter')

    def hsvFilter(self):
        hsvFilt = cv.inRange(self.hsvImage, lowerb=self.lowerThreshold, upperb=self.upperThreshold)
        return hsvFilt

    def findContours(self):
        hsvFilt = self.hsvFilter()
        #   Поиск контуров и сохранение их в переменную contours
        _, contours, __ = cv.findContours(hsvFilt.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        return contours

    def drawAims(self, color, threshold=None, pointMin=None, pointMax=None):
        if color == (255, 0, 0):    # Если фильтруем синих роботов
            numberOfRobots = "Роботов в синей команде: "
            filename = 'BlueRobots.jpg' # имя бинарного изобр., в которое будут сохранены отфильрованные роботы
            print("Фильтруем синих роботов")
        elif color == (0, 255, 0):  # Если фильтруем зеленых роботов
            numberOfRobots = "Роботов в зеленой команде: "
            print("Фильтруем зеленых роботов")
            filename = "GreenRobots.jpg"
        elif color == (0, 0, 255):  # Если фильтруем красных роботов
            numberOfRobots = "Роботов в красной команде: "
            print("Фильтруем красных роботов")
            filename = "RedRobots.jpg"
        else:
            print("Нет такой команды, неопределенное поведение...")
            return -1
        # Филтрация синей и зеленой команд
        if color == (255, 0, 0) or color == (0, 255, 0):
            if threshold is None:
                self.writeThresholdBar()
            else:
                self.writeThreshold(threshold[0], threshold[1], threshold[2], threshold[3], threshold[4], threshold[5])
            hsvFilt = self.eroseAndDilate()
            cv.imwrite(filename, hsvFilt)
        # Фильтрация красной команды (делается по другому т.к. не удалось отфильтровать красных по пороговым значениям)
        else:
            # Загружаем отфильтрованные изображения роботов в бинарном формате
            allRobotsED = cv.imread('AllRobotsED.jpg', cv.IMREAD_GRAYSCALE)  # ED - после применения эрозии и дилатации
            cv.inRange(allRobotsED, lowerb=252, upperb=255, dst=allRobotsED)
            blueRobots = cv.imread('BlueRobots.jpg', cv.IMREAD_GRAYSCALE)
            cv.inRange(blueRobots, lowerb=252, upperb=255, dst=blueRobots)
            greenRobots = cv.imread('GreenRobots.jpg', cv.IMREAD_GRAYSCALE)
            cv.inRange(greenRobots, lowerb=252, upperb=255, dst=greenRobots)

            # Фильтруем красных роботов
            redRobots = cv.subtract(allRobotsED, blueRobots)
            redRobots = cv.subtract(redRobots, greenRobots)
            # self.hsvImage = redRobots.copy()
            hsvFilt = self.eroseAndDilate(redRobots)
            cv.imwrite("RedRobots.jpg", hsvFilt)

        _, contours, __ = cv.findContours(hsvFilt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)

        if pointMin is None and pointMax is None:
            self.writeContourSizeBar()
        else:
            self.contourSize[0] = pointMin
            self.contourSize[1] = pointMax

        self.closestToLight[0] = 5000
        countRobots = 0
        for number in range(len(contours)):
            #   Фильтрация контуров-шумов
            if self.contourSize[0] < contours[number].shape[0] < self.contourSize[1]:
                #   Нахождение центра масс робота
                moments = cv.moments(contours[number])
                self.cgX = int(moments['m10'] / moments['m00'])
                self.cgY = int(moments['m01'] / moments['m00'])

                # вычисление расстояния между центрами масс лампы и робота
                dist = math.sqrt(pow((self.cgX - self.lightCG[0]), 2) + pow((self.cgY - self.lightCG[1]), 2))

                if dist < 50:  # фильтрация контура абажура лампы
                    continue
                # Нахождение ближайшего робота команды к лампе
                if dist < self.closestToLight[0]:
                    self.closestToLight[0] = dist
                    self.closestToLight[1:] = self.cgX, self.cgY
                # обводка крышки робота цветом его команды
                countRobots += 1
                cv.drawContours(image=self.image, contours=contours, contourIdx=number, color=color, thickness=2)
        # обозначение центров масс ближайших роботов из каждой команды
        cv.circle(self.image, (self.closestToLight[1], self.closestToLight[2]), 4, color, -1)
        print(numberOfRobots + str(countRobots))

    def drawThreeCommands(self, blueThresh=None, greenThresh=None, redThresh=None, pointMin=None, pointMax=None):
        # Обозначение синей команды
        self.drawAims(color=(255, 0, 0), threshold=blueThresh, pointMin=pointMin, pointMax=pointMax)
        # Обозначение зеленой команды
        self.drawAims(color=(0, 255, 0), threshold=greenThresh, pointMin=pointMin, pointMax=pointMax)
        # Обозначение красной команды
        self.drawAims(color=(0, 0, 255), threshold=redThresh, pointMin=pointMin, pointMax=pointMax)

    def drawLight(self):
        contours = self.findContours()
        moments = cv.moments(contours[0])
        #   Нахождение центра масс лампы
        self.lightCG[0] = int(moments['m10'] / moments['m00'])
        self.lightCG[1] = int(moments['m01'] / moments['m00'])
        cv.drawContours(image=self.image, contours=contours, contourIdx=-1, color=(0, 150, 255), thickness=2)

    def drawCGContours(self, pointMin=None, pointMax=None):
        if pointMin is None and pointMax is None:
            self.writeContourSizeBar()
        else:
            self.contourSize[0] = pointMin
            self.contourSize[1] = pointMax

        contours = self.findContours()

        for number in range(len(contours)):
            #   Фильтрация контуров-шумов
            if contours[number].shape[0] >= self.contourSize[0] and contours[number].shape[0] < self.contourSize[1]:
                #   Нахождение момента контура
                moments = cv.moments(contours[number])
                #   Нахождение центра масс контура
                self.cgX = int(moments['m10'] / moments['m00'])
                self.cgY = int(moments['m01'] / moments['m00'])
                cv.circle(self.image, (self.cgX, self.cgY), 3, (0, 0, 0), -1)


#   Переход в директорию с изображениями для задания п.1
# os.chdir(pathAllababah)
#
# #   Загрузка изображений для п.1
# image11 = cv.imread('target_1.jpg', cv.IMREAD_COLOR)
# image12 = cv.imread('target_2.jpg', cv.IMREAD_COLOR)
# image13 = cv.imread('target_3.jpg', cv.IMREAD_COLOR)
#
# #   Обозначение центров басурманских домиков для атаки
# #   Рекомендуемые параметры для всех: Hmax = Smax = 0
#
# target_11 = Aims(image11)
# # target_11.writeThresholdBar() # если необходима настройка пороговыйх значений фильтра-заменить этой строкой следующую
# target_11.writeThreshold(vMin=210, hMax=0, sMax=0)
# target_11.drawCGContours(pointMin=45, pointMax=50)  # если необходима настройка - вызвать функцию без аргументов
# # target_12.drawCGContours()
#
# target_12 = Aims(image12)
# # target_11.writeThresholdBar() # если необходима настройка пороговыйх значений фильтра-заменить этой строкой следующую
# target_12.writeThreshold(vMin=220, hMax=0, sMax=0)
# target_12.drawCGContours(pointMin=10, pointMax=50)  # если необходима настройка - вызвать функцию без аргументов
# # target_12.drawCGContours()
#
# target_13 = Aims(image13)
# # target_11.writeThresholdBar() # если необходима настройка пороговыйх значений фильтра-заменить этой строкой следующую
# target_13.writeThreshold(vMin=210, hMax=0, sMax=0)
# target_13.drawCGContours(pointMin=10, pointMax=50)  # если необходима настройка - вызвать функцию без аргументов
# # target_13.drawCGContours()
#
# cv.imshow('target_1', target_11.image)
# cv.imshow('target_2', target_12.image)
# cv.imshow('target_3', target_13.image)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

#   Переход в директорию с изображениями для задания п.2
# os.chdir(pathTeplovizor)
#
# #   Загрузка изображений для п.2
# image21 = cv.imread('firstPlane.jpg', cv.IMREAD_COLOR)
# image22 = cv.imread('secondPlane.jpg', cv.IMREAD_COLOR)
# image23 = cv.imread('ship.png', cv.IMREAD_COLOR)
# image24 = cv.imread('shuttle.jpg', cv.IMREAD_COLOR)
# image25 = cv.imread('turbine.jpg', cv.IMREAD_COLOR)
#
# #   Обозначение лучших мест на басурманских аппаратах для наведения ракет
# firstPlane = Aims(image21)
# # firstPlane.writeThresholdBar() # если необходима настройка пороговыйх значений фильтра-заменить этой строкой следующую
# firstPlane.writeThreshold(hMax=25)  # Рекомендуемые параметры: Hmax = 25
# firstPlane.drawCGContours(pointMin=0, pointMax=50)  # если необходима настройка - вызвать функцию без аргументов
# # firstPlane.drawCGContours()
#
# secondPlane = Aims(image22)
# # secondPlane.writeThresholdBar() # если необходима настройка пороговыйх значений фильтра-заменить этой строкой следующую
# secondPlane.writeThreshold(hMax=13)  #  Рекомендуемые параметры: Hmax = 13 - наведение в хвост
# secondPlane.drawCGContours(pointMin=10, pointMax=50)  # если необходима настройка - вызвать функцию без аргументов
# # secondPlane.drawCGContours()
#
# ship = Aims(image23)
# # ship.writeThresholdBar() # если необходима настройка пороговыйх значений фильтра-заменить этой строкой следующую
# ship.writeThreshold(hMax=16)  #  Рекомендуемые параметры: Hmax = 16(приближенно)
# ship.drawCGContours(pointMin=40, pointMax=70)  # если необходима настройка - вызвать функцию без аргументов
# # ship.drawCGContours()
#
# shuttle = Aims(image24)
# # shuttle.writeThresholdBar() # если необходима настройка пороговыйх значений фильтра-заменить этой строкой следующую
# shuttle.writeThreshold(hMin=18, vMin=187, hMax=36)  #  Рекомендуемые параметры: Hmin = 18, Vmin = 187, Hmax = 36
# shuttle.drawCGContours(pointMin=10, pointMax=50)  # если необходима настройка - вызвать функцию без аргументов
# # shuttle.drawCGContours()
#
# turbine = Aims(image25)
# # turbine.writeThresholdBar() # если необходима настройка пороговыйх значений фильтра-заменить этой строкой следующую
# turbine.writeThreshold(vMin=56, hMax=31, sMax=82)  #  Рекомендуемые параметры: Vmin = 56; Hmax = 31, Smax = 82
# turbine.drawCGContours(pointMin=10, pointMax=50)  # если необходима настройка - вызвать функцию без аргументов
# # turbine.drawCGContours()
#
# #   Отображение моторных отделений на забугорных аппаратах для наведения ракеты
# cv.imshow('firstPlane', firstPlane.image)
# cv.imshow('secondPlane', secondPlane.image)
# cv.imshow('ship', ship.image)
# cv.imshow('shuttle', shuttle.image)
# cv.imshow('turbine', turbine.image)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

# Переход в директорию с изображениями для задания п.3
os.chdir(pathRoboti)
object1 = cv.imread('yellow_object_1.jpg', cv.IMREAD_COLOR)
object2 = cv.imread('yellow_object_2.jpg', cv.IMREAD_COLOR)
object3 = cv.imread('red_object_1.jpg', cv.IMREAD_COLOR)
object4 = cv.imread('red_object_2.jpg', cv.IMREAD_COLOR)
object5 = cv.imread('color_object_1.jpg', cv.IMREAD_COLOR)
object6 = cv.imread('color_object_2.jpg', cv.IMREAD_COLOR)

object_1 = Aims(object4)
object_1.writeThresholdBar()
# #   Загрузка изображений для п.3
# roiRobotov1 = cv.imread('roi_robotov_1.jpg', cv.IMREAD_COLOR)
# roiRobotov2 = cv.imread('roi_robotov_2.jpg', cv.IMREAD_COLOR)
#
# blueThreshold = [87, 49, 113, 159, 255, 255]
# greenThreshold = [65, 47, 133, 77, 255, 255]
# allRobotsThreshold = [0, 59, 134, 179, 255, 255]
#
# # Фильтрация ВСЕХ роботов! - необходима для дальнейшней фильтрации ТОЛЬКО красных роботов
# allRobots = cv.cvtColor(roiRobotov1, cv.COLOR_BGR2HSV)
# allRobots = cv.inRange(allRobots, (allRobotsThreshold[0], allRobotsThreshold[1], allRobotsThreshold[2]),
#                        (allRobotsThreshold[3], allRobotsThreshold[4], allRobotsThreshold[5]))
# allRobots = Aims.eroseAndDilate(0, image=allRobots)
# cv.imwrite('AllRobotsED.jpg', allRobots)
#
# robot1 = Aims(roiRobotov1)
# # robot1.writeThresholdBar()
# robot1.writeThreshold(vMin=252, sMax=9)   # Фильтр лампы
# robot1.drawLight()
# robot1.drawThreeCommands(blueThresh=blueThreshold, greenThresh=greenThreshold, pointMin=10, pointMax=50)
#
# cv.imshow('Roi robotov 1', robot1.image)
# cv.imwrite('AllRobotsIsdrawed1.jpg', robot1.image)
# cv.waitKey(0)
# cv.destroyWindow('Roi robotov 1')
#
# # Фильтрация ВСЕХ роботов! - необходима для дальнейшней фильтрации ТОЛЬКО красных роботов
# allRobots = cv.cvtColor(roiRobotov2, cv.COLOR_BGR2HSV)
# allRobots = cv.inRange(allRobots, (allRobotsThreshold[0], allRobotsThreshold[1], allRobotsThreshold[2]),
#                        (allRobotsThreshold[3], allRobotsThreshold[4], allRobotsThreshold[5]))
# allRobots = Aims.eroseAndDilate(0, image=allRobots)
# cv.imwrite('AllRobotsED.jpg', allRobots)
#
# robot2 = Aims(roiRobotov2)
# # robot2.writeThresholdBar()
# robot2.writeThreshold(vMin=252, sMax=9)   # Фильтр лампы
# robot2.drawLight()  # Обозначаем лампу
# robot2.drawThreeCommands(blueThresh=blueThreshold, greenThresh=greenThreshold, pointMin=10, pointMax=50)
#
# cv.imshow('Roi robotov 2', robot2.image)
# cv.imwrite('AllRobotsIsdrawed2.jpg', robot2.image)
# cv.waitKey(0)
# cv.destroyWindow('Roi robotov 2')

#   Переход в директорию с изображениями для задания п.4
# os.chdir(pathGk)
#
# #   Загрузка изображений для п.4
# image41 = cv.imread('gk.jpg', cv.IMREAD_COLOR)
# image42 = cv.imread('gk_tmplt.jpg', cv.IMREAD_COLOR)
#
# gkSample = Aims(image42)
# contGkSample = gkSample.findContours()
# gkSample.drawCGContours(pointMin=0, pointMax=300)
#
# gks = Aims(image41)
# # gk.writeThresholdBar()    # раскомментировать, если необходимо определить пороговые значения экспериментально
# gks.writeThreshold(vMax=239) # а эту тогда закомментировать, если верхняя строка раскомментирована
# # gk.drawAims(pointMin=16, pointMax=41)
# contGks = gks.findContours()
# """ Что ниже - лучше бы в новый метод класса поместить (мне было лень)"""
# for number in range(len(contGks)):
#     r = cv.matchShapes(contour1=contGkSample[0], contour2=contGks[number], method=cv.CONTOURS_MATCH_I1, parameter=0)
#     if r > 10:  # поч десять спросите вы, да потому что определено экспериментально
#         color = (0, 0, 255) # годные ключи помечаются красной меткой
#     else:
#         color = (255, 0, 0) # бракованные ключи помечаются синей меткой
#     moments = cv.moments(contGks[number])
#     #   Нахождение центра масс робота
#     cgX = int(moments['m10'] / moments['m00'])
#     cgY = int(moments['m01'] / moments['m00'])
#     cv.circle(gks.image, (cgX, cgY), 4, color, -1)
#     print(r)
#
# cv.imshow('gks', gks.image)
# cv.waitKey(0)
# cv.destroyWindow('gks')
