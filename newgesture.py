import cv2
import numpy as np
from math import sqrt, fabs, pow, acos, pi
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

dedos = np.zeros((5,5,4))
dedos_cont = 0;

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

def angulo(s,f,e):
    l1 = distancia(f,s)
    l2 = distancia(f,e)
    ponto = (s[0] - f[0]) * (e[0] - f[0]) + (s[1] - f[1]) * (e[1] - f[1])
    try:
        angulo = acos(ponto/(l1*l2))
    except ValueError:
        return 0    
    angulo = (angulo*180)/pi
    return angulo

if __name__ == "__main__":
    extrator = cv2.createBackgroundSubtractorMOG2() #extrator do cenario
    extrator.setDetectShadows(False) #nao detectar as sombras
    cont = 0 #contador de frames
    pause = False
    cx = 0
    cy = 0
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
            img = cap.read()[1]
            
            #b,g,r = cv2.split(img)
            gray = extrator.apply(img,None,0)
            #blur = cv2.medianBlur(gray, 5)
            try:
                contours, hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            except:
                break
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
                cv2.drawContours(img,[cnt],0,(0,255,0),2)
                hull = cv2.approxPolyDP(hull,epsilon=13,closed=True)
                
                cv2.drawContours(img,[hull],0,(0,0,255),2)
                j = 0
                d_maior = 0
                maior_dedo = hull[0]
                for i in hull:
                    if i[0][1] < 400:
                        dedos[dedos_cont,j,0] = i[0][0]
                        dedos[dedos_cont,j,1] = i[0][1]
                        cv2.line(img, (i[0][0],i[0][1]), (cx,cy), azul, 2)
                        d = distancia((i[0][0],i[0][1]), (cx,cy))
                        j = j + 1
                        if d > d_maior:
                            maior_dedo = i
                            d_maior = d
                
                ###############
                

                for i in range(0,5):
                    dedos[dedos_cont,j,2] = angulo((dedos[dedos_cont,1,0],dedos[dedos_cont,1,1]),centr,maior_dedo[0])

                for i in hull:
                    if i[0][1] < 400:
                        inicio = (i[0][0],i[0][1]) 
                        if(angulo(inicio,centr,maior_dedo[0])>20 and angulo(inicio,centr,maior_dedo[0])<35 and inicio[0]<maior_dedo[0][0]):
                            cv2.putText(img,'anelar', (i[0][0],i[0][1] + 40),1,1,branco)
                        elif(angulo(inicio,centr,maior_dedo[0])>40 and angulo(inicio,centr,maior_dedo[0])<55 and inicio[0]<maior_dedo[0][0]):
                            cv2.putText(img,'minimo', (i[0][0],i[0][1] + 40),1,1,branco)
                        elif(angulo(inicio,centr,maior_dedo[0])==0):
                            cv2.putText(img,'medio', (i[0][0],i[0][1] + 40),1,1,branco)
                        elif(angulo(inicio,centr,maior_dedo[0])>20 and angulo(inicio,centr,maior_dedo[0])<35 and inicio[0]>maior_dedo[0][0]):
                            cv2.putText(img,'indicador', (i[0][0],i[0][1] + 40),1,1,branco)
                        elif(angulo(inicio,centr,maior_dedo[0])>40 and angulo(inicio,centr,maior_dedo[0])<120 and inicio[0]>maior_dedo[0][0]):
                            cv2.putText(img,'polegar', (i[0][0],i[0][1] + 40),1,1,branco)
                
                dedos_cont +=1
                if(dedos_cont>=5):
                    dedos_cont = 0
                ################

                cv2.imshow('input',img)                   

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k == 32:
            pause = not pause
            cv2.putText(img, "PAUSE", (10, 40), 1, 2, branco)
            cv2.imshow("input", img)
