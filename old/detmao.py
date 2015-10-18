#!/usr/bin/python
import sys, os, argparse
import pickle
import numpy as np
import cv2

'''
CADA NUMERO REPRESENTA UM DEDO
menor       =   1
anelar      =   2
medio       =   3
indicador   =   4
polegar     =   5
'''

parser = argparse.ArgumentParser()
parser.add_argument("-nf", "--frames", type=int, default=40)
parser.add_argument("arquivo", nargs="?", default=None)
args = parser.parse_args()

if args.arquivo:
    print "Capturando a partir do arquivo..."
    cap = cv2.VideoCapture(args.arquivo)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        print "Erro: nao foi possivel abrir o arquivo"
        sys.exit(1)
else:
    print "Capturando a partir da camera..."
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print "Erro: nenhuma camera localizada"
        sys.exit(1)

# CORES
verde    = [0, 255, 0]
vermelho = [0, 0, 255]
branco   = [255, 255, 255]

# EXTRATOR DE CENARIO
extrator = cv2.createBackgroundSubtractorMOG2()
extrator.setDetectShadows(False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #Matriz para a filtragem

def main():
    fps = cap.get(cv2.CAP_PROP_FPS)
    tempo_entre_frames = int(1000/fps)
    ok = True
    pause = False
    coletaBackground()
    while ok:
        if not pause:
            ok, frame = cap.read()
            if not ok:
                break
            mascara = extrator.apply(frame, None, 0)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel) #Filtro para suavizar o ruido
            # mascara = cv2.medianBlur(mascara, 5)  # Filtros para suavizar o ruido
            detectaMao(frame,mascara)
        frame_atual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        k = cv2.waitKey(tempo_entre_frames) & 0xFF
        if k == 27:  # Sair ao pressionar esc
            print "Saindo..."
            break
        elif k == 32: # Pausar ao pressionar espaco
            pause = not pause
            if pause:
                print "PAUSE"
            else:
                print "PLAY"
        elif k == 81 and frame_atual > fps:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_atual-fps)
            print "-1 segundo"
        elif k == 83 and frame_atual + fps < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_atual+fps)
            print "+1 segundo"
    print "Fim"
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

def coletaBackground():
    for i in xrange(1,args.frames+1):
        ok, frame = cap.read()
        mascara = extrator.apply(frame)
        sys.stdout.write("\rCriando amostra do cenario: %d%%" % (100 * i / args.frames))  # Progresso do aprendizado
        sys.stdout.flush()
    print "\nConcluido"

def selecionaMaiorContorno(cnt):
    maior_contorno = cnt[0]
    for c in cnt:
        if len(c) > len(maior_contorno):
            maior_contorno = c
    return maior_contorno

def calculaAngulo(a,origem,b):
    A = np.subtract(a,origem)
    B = np.subtract(b,origem)
    dot = np.dot(A,B)
    modA = np.linalg.norm(A)
    modB = np.linalg.norm(B)
    angulo = np.arccos(dot/(modA*modB))
    angulo = np.degrees(angulo)
    return angulo

def detectaMao(frame, mascara):
    bckpmas = np.copy(mascara)
    img, contornos, hieraquia = cv2.findContours(mascara, 0, 2)
    if len(contornos) == 0:
        return
    maior_contorno = selecionaMaiorContorno(contornos)
    cv2.polylines(frame, [maior_contorno], True, verde, 1)
    hull = cv2.convexHull(maior_contorno, returnPoints=False)
    defects = cv2.convexityDefects(maior_contorno, hull)
    if defects == None:
        return
    defects_detectados = [] # Lista vazia de defects detectados
    for i in xrange(defects.shape[0]):
        a, b, c, d = defects[i, 0]
        ini   = tuple(maior_contorno[a, 0])
        fim   = tuple(maior_contorno[b, 0])
        cvx   = tuple(maior_contorno[c, 0])
        cv2.line(frame,ini,fim,verde,1)
        angulo  = calculaAngulo(ini,cvx,fim)
        if 10 < angulo < 120 and d > 15000:
            pos = c*100/len(maior_contorno) # posicao do defect no contorno
            dados = (ini,cvx,fim,angulo,pos)
            defects_detectados.append(dados)
            cv2.putText(frame,str(pos),cvx,1,1,branco)
            cv2.circle(frame,cvx,3,vermelho,-1)
    dedos = identificaDedos(defects_detectados)
    exibeInfo(frame,bckpmas,dedos)

def identificaDedos(defects_detectados):
    num = len(defects_detectados)
    dedos = set([]) # acumulador do tipo set para os dedos
    if num == 1:
        if defects_detectados[0][3] < 80:
            if defects_detectados[0][4] > 80:
                dedos.update([3,4])
            else:
                dedos.update([1,2])
        else:
            dedos.add(4)
    elif num == 2:
        for d in defects_detectados:
            inicio,conv,fim,angulo,pos = d
            if 80 < angulo < 100 and pos < 90: # angulo entre o polegar e o indicador
                dedos.update([4,5])
            elif angulo <= 80:
                if pos < 20:
                    dedos.update([2,3])
                elif pos < 40:
                    dedos.update([1,2])
                else:
                    dedos.update([3,4])
    else:
        for d in defects_detectados:
            inicio,conv,fim,angulo,pos = d
            if angulo < 100:
                if pos < 15:
                    dedos.update([3,2])
                elif pos < 45:
                    dedos.update([1,2])
                elif pos < 85:
                    dedos.update([5,4])
                else:
                    dedos.update([3,4])
    return dedos

def exibeInfo(frame,mascara,dedos):
    tamanho = (frame.shape[1]/2,frame.shape[0]/2)
    resized = cv2.resize(mascara,tamanho)
    resized = cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR) # Converter de cinza para RGB antes de concatenar com o frame
    infos = np.zeros(resized.shape,np.uint8) # Cria uma imagem vazia para a informacao
    infos[:] = [94, 68, 55] # Muda a cor de fundo da tela de informacao
    cv2.putText(infos,"Detectados:",(20,40),1,1.8,verde)
    nomes = ["menor","anelar","medio","indicador","polegar"]
    for d in dedos:
        cv2.putText(infos,nomes[d-1],(20,40+d*30),1,1.4,verde) # imprime linhas de informacao
    lateral = np.concatenate([resized,infos],0) # concatena as imagens na lateral
    combImg = np.concatenate([frame,lateral],1) # concatena a lateral com o frame
    cv2.imshow("Detector de Maos",combImg)

if __name__ == "__main__":
    main()
