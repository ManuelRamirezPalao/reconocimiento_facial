import cv2
import face_recognition
imagen_personal = face_recognition.load_image_file(r"C:\Users\manue\Downloads\foto_personal.jpg")
imagen_Brisa = face_recognition.load_image_file(r"C:\Users\manue\Downloads\Brisa.jpeg")
imagen_Dona = face_recognition.load_image_file(r"C:\Users\manue\Downloads\Donaji.jpeg")
imagen_Ale = face_recognition.load_image_file(r"C:\Users\manue\Downloads\Ale.jpeg")
imagen_coco = face_recognition.load_image_file(r"C:\Users\manue\Downloads\Socorro.jpeg")
imagen_Silva = face_recognition.load_image_file(r"C:\Users\manue\Downloads\Silva.jpeg")

personal_encodings = face_recognition.face_encodings(imagen_personal)[0]
brisa_encondings = face_recognition.face_encodings(imagen_Brisa)[0]

dona_encodings = face_recognition.face_encodings(imagen_Dona)[0]
ale_encodings = face_recognition.face_encodings(imagen_Ale)[0]
coco_encodings = face_recognition.face_encodings(imagen_coco)[0]
silva_encodings = face_recognition.face_encodings(imagen_Silva)[0]




encodings_conocidos = [
    personal_encodings,
    brisa_encondings,
    dona_encodings,
    ale_encodings,
    coco_encodings,
    silva_encodings
]
nombres_conocidos = [
    "Manuel Ramirez Palao",
    "Brisa Quiroz Rangel",
    "Donaji Sanchez Nieves",
    "Alejandra Figueroa",
    "Coco Palao",
    "Silva Luna"
]

#Iniciar la webcam:
webcam = cv2.VideoCapture(1)

#Cargar una fuente de texto:
font = cv2.FONT_HERSHEY_COMPLEX


reduccion = 5

print("\nRecordatorio: pulsa 'ESC' para cerrar.\n")


while 1:

    loc_rostros = []
    encodings_rostros = []
    nombres_rostros = []
    nombre = ""

    #Capturamos una imagen con la webcam:
    valido, img = webcam.read()


    if valido:

        #La imagen está en el espacio de color BGR, habitual de OpenCV. Hay que convertirla a RGB:
        img_rgb = img[:, :, ::-1]

        #Reducimos el tamaño de la imagen para que sea más rápida de procesar:
        img_rgb = cv2.resize(img_rgb, (0, 0), fx=1.0/reduccion, fy=1.0/reduccion)

        #Localizamos cada rostro de la imagen y extraemos sus encodings:
        loc_rostros = face_recognition.face_locations(img_rgb)
        encodings_rostros = face_recognition.face_encodings(img_rgb, loc_rostros)



        for encoding in encodings_rostros:
            coincidencias = face_recognition.compare_faces(encodings_conocidos, encoding)


            if True in coincidencias:
                nombre = nombres_conocidos[coincidencias.index(True)]

            else:
                nombre = "No reconocido"

            #Añadir el nombre de la persona identificada en el array de nombres:
            nombres_rostros.append(nombre)


        for (top, right, bottom, left), nombre in zip(loc_rostros, nombres_rostros):

            #Deshacemos la reducción de tamaño para tener las coordenadas de la imagen original:
            top = top*reduccion
            right = right*reduccion
            bottom = bottom*reduccion
            left = left*reduccion

            #Cambiar de color según si se ha identificado el rostro:
            if nombre != "No reconocido":
                color = (0,255,0)
            else:
                color = (0,0,255)

            #Dibujar un rectángulo alrededor de cada rostro identificado, y escribir el nombre:
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.rectangle(img, (left, bottom - 20), (right, bottom), color, -1)
            cv2.putText(img, nombre, (left, bottom - 6), font, 0.6, (0,0,0), 1)


        cv2.imshow('Output', img)

        #Salir con 'ESC'
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

webcam.release()
