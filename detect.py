#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import sys

class Dedo:
	posicao = (0, 0)
	tamanho = 0
	angulo  = 0
	detectado = False

parser = argparse.ArgumentParser()
parser.add_argument("-nf","--frames",type=int,default=30) #Numero de Frames para aprender o cenario
parser.add_argument("arquivo",nargs="?",default=None) #Nome do arquivo, se houver
parser.add_argument("-o","--output", nargs="?",default=None)
args = parser.parse_args()

# CORES
VERDE = [0, 255, 0]
VERMELHO = [0, 0, 255]
AZUL = [255, 0, 0]
BRANCO = [255, 255, 255]

centro = (0, 0)
medio = Dedo()
anelar = Dedo()
minimo = Dedo()
polegar = Dedo()
indicador = Dedo()

dedos_detectados = [minimo, anelar, medio, indicador, polegar]
dedos_menor_tam  = {"minimo": None, "anelar": None, "medio": None, "indicador": None, "polegar": None}

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

framerate = cap.get(cv2.CAP_PROP_FPS)
resolucao = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if args.output:
	fourcc = cv2.VideoWriter_fourcc(*"XVID")
	out = cv2.VideoWriter(args.output+".avi",fourcc, framerate, resolucao)
	print "Salvando arquivo em",args.output+".avi"

def distancia(ponto, origem):
	p = np.subtract(ponto,origem)
	return np.linalg.norm(p)

def angulo(s,f,e):
	l1 = distancia(f,s)
	l2 = distancia(f,e)
	if l1 * l2 == 0:
		return np.nan
	dot = np.dot(np.subtract(f, s), np.subtract(f, e))
	div = dot / (l1*l2)
	if div > 1:
		div = 1
	elif div < -1:
		div = -1
	angulo = np.degrees(np.arccos(div))
	return angulo 

def capturaCenario(numeroDeFrames):
	extrator = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
	for i in range(0, numeroDeFrames):
		ok, frame = cap.read()
		if not ok:
			print "Houve um erro ao tentar capturar a amostra do cenário."
			sys.exit(1)
		mascara = extrator.apply(frame) #Aplica o extrator com a taxa maxima de aprendizado
		percent = (100 * (i + 1) / numeroDeFrames)
		sys.stdout.write("\rCriando amostra do cenÃ¡rio: %d%%" % percent) #Progresso do aprendizado
		sys.stdout.flush()
	print "\nConcluÃ­do."
	return extrator

def detectaContorno(frame):
	contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	if len(contours) == 0:
		return []
	cnt = contours[0]
	for c in contours:
		if (len(c) > len(cnt)):
			cnt = c
	return cnt

def detectaCentro(contorno):
	cx = 0
	cy = 0
	moments = cv2.moments(contorno)
	if moments['m00']!=0:
		cx = int(moments['m10'] / moments['m00']) # cx = M10/M00
		cy = int(moments['m01'] / moments['m00']) # cy = M01/M00
	return (cx, cy)

def detectaApprox(contorno):
	hull = cv2.convexHull(contorno)
	approx = cv2.approxPolyDP(hull, epsilon = 13, closed = True)
	return approx

def detectaMaiorDedo(approx):
	maior_tam = 0
	menor_x = approx[0][0][0]
	for ponto in approx:
		x = ponto[0][0]
		y = ponto[0][1]
		ang_vertical = angulo((x, y), centro, (centro[0], 0))
		if y < 400 and ang_vertical < 30:
			tam = distancia((x, y), centro)
			if tam > maior_tam:
				maior_tam = tam
				medio.tamanho = tam
				medio.posicao = (x, y)
				medio.detectado = True
			if x < menor_x:
				menor_x = x
				minimo.tamanho = tam
				minimo.posicao = (x, y)
				minimo.detectado = True

def detectaDedos(approx):
	for dedo in dedos_detectados:
		dedo.detectado = False
	detectaMaiorDedo(approx)
	for ponto in approx:
		pos = (ponto[0][0], ponto[0][1])
		tam = distancia(pos, centro)
		ang = angulo(pos, centro, medio.posicao)
		dedo = identificaDedo(ang, pos[0])
		if dedo != None:
			dedo.detectado = True
			dedo.posicao = pos
			dedo.tamanho = tam
			dedo.angulo = ang
	for dedo in dedos_detectados:
		if not dedo.detectado:
			pos = estimaPosicao(dedo, approx)
			dedo.posicao = pos

def exibeInfo(frame, centro, contorno, approx, dedos_detectados):
	cv2.circle(frame, centro, 5, VERMELHO, -2)
	cv2.drawContours(frame, [cnt], 0, VERDE, 2)
	cv2.drawContours(frame,[approx], 0, AZUL, 2)
	cv2.circle(frame, medio.posicao, 5, VERMELHO, -2)
	for dedo in dedos_detectados:
		cv2.putText(frame, nomeDedo(dedo), dedo.posicao, 1, 1, BRANCO)
		cv2.line(frame, dedo.posicao, centro, AZUL, 2)

def nomeDedo(dedo):
	if dedo == polegar:
		return "polegar"
	if dedo == indicador:
		return "indicador"
	if dedo == medio:
		return "medio"
	if dedo == anelar:
		return "anelar"
	if dedo == minimo:
		return "minimo"

def identificaDedo(angulo, x):
	if angulo == 0:
		return medio
	elif angulo > 20 and angulo < 35 and x < medio.posicao[0] :
		return anelar
	elif angulo > 40 and angulo < 65 and x < medio.posicao[0] :
		return minimo
	elif angulo > 20 and angulo < 35 and x > medio.posicao[0] :
		return indicador
	elif angulo > 40 and angulo < 120 and x > medio.posicao[0] :
		return polegar
	else:
		return None

def pontoMedio(ponto_a, ponto_b):
	x = (ponto_a[0] + ponto_b[0]) / 2
	y = (ponto_a[1] + ponto_b[1]) / 2
	return (int(x), int(y))

def estimaPosicao(dedo, approx):
	tamanho = medio.tamanho
	x = 0
	y = 0
	angulo = 0
	if dedo == anelar:
		angulo = np.radians(-30)
	if dedo == minimo:
		angulo = np.radians(-50)
	if dedo == indicador:
		angulo = np.radians(40)
	if dedo == polegar:
		angulo = np.radians(100)
	x = centro[0] + np.sin(angulo) * tamanho
	y = centro[1] - np.cos(angulo) * tamanho
	return (int(x), int(y))


if __name__ == "__main__":
	extrator = capturaCenario(args.frames)
	pause = False

	while(cap.isOpened()):
		if not pause:
			ok, frame = cap.read()
			if not ok:
				break
			grey = extrator.apply(frame, None, 0)
			
			cnt = detectaContorno(grey)
			if len(cnt) == 0:
				continue
			centro = detectaCentro(cnt)
			approx = detectaApprox(cnt)
			detectaDedos(approx)
			exibeInfo(frame, centro, cnt, approx, dedos_detectados)

		if args.output:
			out.write(frame)
			
		cv2.imshow('input', frame)

		k = cv2.waitKey(30) & 0xFF
		#k = cv2.waitKey(int(1000/framerate)) & 0xFF
		if k == 27:
			break
		elif k == 32:
			pause = not pause
			cv2.putText(frame, "PAUSE", (10, 40), 1, 2, BRANCO)
			cv2.imshow("input", frame)
		elif k == 102:
		    for dedo in dedos_detectados:
		        nome = nomeDedo(dedo)
		        dedos_menor_tam[nome] = dedo.tamanho
		if not dedos_menor_tam["anelar"] == None:  
		    print str(dedos_menor_tam["anelar"] - anelar.tamanho)
