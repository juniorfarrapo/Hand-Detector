import cv2
import numpy as np
import argparse
import sys


# CORES
verde = [0, 255, 0]
ciano = [255, 255, 0]
vermelho = [0, 0, 255]
azul = [255, 0, 0]
amarelo = [0, 255, 255]
preto = [0, 0, 0]
branco = [255, 255, 255]

parser = argparse.ArgumentParser()
parser.add_argument("-nf","--frames",type=int,default=30) #Numero de Frames para aprender o cenario
parser.add_argument("arquivo",nargs="?",default=None) #Nome do arquivo, se houver
args = parser.parse_args()

if args.arquivo:
    print "Capturando a partir do arquivo..."
    cap = cv2.VideoCapture(args.arquivo)
    if not cap.isOpened():
        print "Erro: nao foi possivel abrir o arquivo"
        sys.exit(1)
else:
    print "Capturando a partir da camera..."
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print "Erro: nenhuma camera localizada"
        sys.exit(1)

def calculaEixos(centro,ext):
    Y = np.subtract(centro,ext)
    X = np.rot90(Y)
    x1 = (centro[0]+X[0],centro[1]-X[1])
    x2 = (centro[0]-X[0],centro[1]+X[1])
    y1 = tuple(ext[0])
    y2 = (centro[0]+Y[0][0],centro[1]+Y[0][1])
    cv2.line(img,x1,x2,amarelo,3)
    cv2.line(img,y1,y2,amarelo,3)
    cv2.circle(img,centro,int(np.linalg.norm(Y)),amarelo,3)

def distancia(a,origem):
    A = np.subtract(a,origem)
    modA = np.linalg.norm(A)
    return modA

if __name__ == "__main__":
    extrator = cv2.createBackgroundSubtractorMOG2() #extrator do cenario
    extrator.setDetectShadows(False) #nao detectar as sombras
    cont = 0 #contador de frames
    pause = False

    while(cont<args.frames):
        cont += 1;
        ret, frame = cap.read()
        mascara = extrator.apply(frame) #Aplica o extrator com a taxa maxima de aprendizado
        #cv2.imshow("teste",mascara)
        sys.stdout.write("\rCriando amostra do cenario: %d%%" % (100*cont/args.frames) ) #Progresso do aprendizado
        sys.stdout.flush()
        if cont == args.frames:
            print "\nConcluido."


    while(cap.isOpened()):
        #recebe imagem, transforma cinza, aplica blur, aplica limiarizacao, busca por contornos e cria uma matriz de zeros
        if not pause:
            ok,img = cap.read()
            if not ok:
                break
            gray = extrator.apply(img,None,0)
            blur = cv2.medianBlur(gray, 5)
            contours, hierarchy = cv2.findContours(blur,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            if(len(contours) > 0):
                cnt = contours[0]
                #busca maior contorno e salva em cnt
                for i in contours:
                        if(len(i)>len(cnt)):
                            cnt=i

                #gera a envoltoria convexa
                hull = cv2.convexHull(cnt)

                #define o centro do contorno
                moments = cv2.moments(cnt)
                if moments['m00']!=0:
                            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                            cy = int(moments['m01']/moments['m00']) # cy = M01/M00

                #aplica o centro do contorno maior na imagem original com um circulo
                centr=(cx,cy)
                cv2.circle(img,centr,5,[0,0,255],2)

                #desenha o contorno maior em verde na matriz de zeros
                cv2.drawContours(img,[cnt],0,verde,2)
                hull = cv2.approxPolyDP(hull,epsilon=13,closed=True)

                cv2.drawContours(img,[hull],0,verde,2)

                d_maior = 0
                maior_dedo = hull[0]
                for i in hull:
                    if i[0][1] < 400:
                        ponto = (i[0][0],i[0][1])
                        d = distancia(ponto,centr)
                        if d > d_maior:
                            maior_dedo = i
                            d_maior = d
                        cv2.line(img, centr, ponto, azul, 1)
                calculaEixos(centr,maior_dedo)

                cv2.imshow('input',img)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k == 32:
            pause = not pause
            cv2.putText(img, "PAUSE", (10, 40), 1, 2, branco)
            cv2.imshow("input", img)
