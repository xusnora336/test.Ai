import cv2


def face_circle():
    # Инициализация классификатора для распознавания лиц
    # haarcascade_frontalface_default.xml - это предобученная модель для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Инициализация видеопотока с камеры (0 - индекс камеры по умолчанию)
    cam = cv2.VideoCapture(0)

    # Бесконечный цикл для обработки каждого кадра
    while True:
        # Чтение кадра с камеры
        # _ - флаг успешного чтения кадра (True/False)
        # frame - сам кадр
        _, frame = cam.read()

        # Преобразование кадра в оттенки серого (упрощает обработку)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на кадре
        # 1.1 - масштабный коэффициент (насколько уменьшается изображение при каждом масштабе)
        # 4 - минимальное количество соседей, необходимое для удержания прямоугольника
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Обработка каждого обнаруженного лица
        for (x, y, w, h) in faces:
            # Вычисление центра лица
            center = (x + w // 2, y + h // 2)
            # Вычисление радиуса окружности (половина ширины лица)
            radius = w // 2
            # Рисование синей окружности вокруг лица
            # (255, 0, 0) - цвет в формате BGR (синий, зеленый, красный)
            # 2 - толщина линии
            cv2.circle(frame, center, radius, (255, 0, 0), 2)

        # Отображение кадра с обведенными лицами
        cv2.imshow("Face Circle", frame)

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов камеры
    cam.release()
    # Закрытие всех окон OpenCV
    cv2.destroyAllWindows()

face_circle()
