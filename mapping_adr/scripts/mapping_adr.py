#!/usr/bin/env python
# Mapping 3D de 3 balizas con quadrotor Hector
import rospy
from turtlesim.msg import Pose as PoseRobot
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray,Pose,PoseStamped, Point32, PointStamped, Point
from sensor_msgs.msg import PointCloud
import tf.transformations
import numpy as np
import random
from math import sqrt,exp
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class Particle:              
    """ Se define la clase de una particula"""

    def __init__(self):
	"""Se inicializa a coordenadas 0"""

        self.x = 0.0
        self.y = 0.0
	self.z = 0.0

class ParticleFilter:
    """ Se define la clase de un filtro"""

    #Atributos de la clase, F distingue el numero del filtro, y baliza las coordenadas GT de la baliza a mapear
    F=0.0
    baliza = [[0.0, 0.0, 0.0]]

    def __init__(self,F,baliza):
	"""Se inicializa la clase"""

        # A subscriber to the topic '/ground_truth_to_tf/pose'. self.update_pose is called
        # when a message of type PoseStamped is received.
        self.pose_subscriber = rospy.Subscriber('/ground_truth_to_tf/pose',
                                                PoseStamped, self.update_pose)

        # A publisher to the topic 'particlesN' and 'balizaN' to publish the cloud of particles and markers
	# in function of filter number
	if F==1:
        	self.particles_publisher = rospy.Publisher('particles1', PointCloud, queue_size = 1)
		self.balizas_publisher = rospy.Publisher('baliza1', PointStamped, queue_size = 1)
		self.f = 1
	if F==2:
        	self.particles_publisher = rospy.Publisher('particles2', PointCloud, queue_size = 1)
		self.balizas_publisher = rospy.Publisher('baliza2', PointStamped, queue_size = 1)
		self.f = 2
	if F==3:
        	self.particles_publisher = rospy.Publisher('particles3', PointCloud, queue_size = 1)
		self.balizas_publisher = rospy.Publisher('baliza3', PointStamped, queue_size = 1)
		self.f = 3
	

        self.odom = Odometry()  
	self.odom_prev = Odometry()      
	self.pose = self.odom.pose.pose.position       #Posicion real del robot
        self.pose_prev = self.odom.pose.pose.position
        self.pose_received = False
	self.pose_listx = []				#Listas para representacion de posicion
	self.pose_listy = []
	self.pose_listz = []
	self.pose_listx.append(self.pose.x)
	self.pose_listy.append(self.pose.y)
	self.pose_listz.append(self.pose.z)

        self.num_particles = 2000
        self.particles = []
        self.weights = []

        self.dist_noise = 0.1
        self.sense_noise = 0.1

	self.d_max = 20.0
        
        self.r_min = -5.0
        self.r_max = 5.0

        self.initialize(baliza)

       
    def update_pose(self, data):
        """Callback function which is called when a new message of type PoseStamped is
        received by the subscriber."""
        
        self.pose.x = round(data.pose.position.x, 4)
        self.pose.y = round(data.pose.position.y, 4)
	self.pose.z = round(data.pose.position.z, 4)
        
        self.odom.pose.pose.orientation.x = data.pose.orientation.x
	self.odom.pose.pose.orientation.y = data.pose.orientation.y
	self.odom.pose.pose.orientation.z = data.pose.orientation.z
	self.odom.pose.pose.orientation.w = data.pose.orientation.w

        if not self.pose_received:
            
            self.pose_prev = copy.deepcopy(self.pose)
	    self.odom_prev = copy.deepcopy(self.odom)
            self.pose_received = True


    def initialize(self,baliza):
        """Function to initialize particles in filter."""

	while True:
		z = self.sense(baliza)       #Se toma medida inicial, para distribuir las particulas
		if z[0]>=0:
			break

        self.particles = []

        for _ in range(self.num_particles):
            p = Particle()
            weight = 0.0

	    phi = 2*np.pi*random.random()
	    theta = 2*np.pi*random.random()

	    #Una vez tomada la primera medida, se distribuyen a esa distancia del robot
	    p.x = z[0]*np.sin(phi)*np.cos(theta)
	    p.y = z[0]*np.sin(phi)*np.sin(theta)
	    p.z = z[0]*np.cos(phi)
	    if p.z < 0:
	    	p.z = -p.z  #Eje Z positivo


	    weight = 1/self.num_particles 

            
            self.particles.append(p)
            self.weights.append(weight)

    def check_robot_motion(self):
        """Function to determine if robot moved enough to update filter.""" #Por si requiere ejecutar paso a paso

        if not self.pose_received:
            return False

        # Compute movement increment from last time
        self.delta_x = self.pose.x - self.pose_prev.x
        self.delta_y = self.pose.y - self.pose_prev.y
	self.delta_z = self.pose.z - self.pose_prev.z

        # Predict only if the robot moved enough
        if abs(self.delta_x) > 0.2 or abs(self.delta_y) > 0.2 or abs(self.delta_z) > 0.2:
            return True
        else:
            return False


    def sense(self,landmarks):
        """Function to generate sensor observation from measuring distange to landmarks."""

        #Z = []
	
	self.pose_listx.append(self.pose.x)
	self.pose_listy.append(self.pose.y)
	self.pose_listz.append(self.pose.z)

        dist = sqrt((self.pose.x - landmarks[0]) ** 2 + (self.pose.y - landmarks[1]) ** 2 + (self.pose.z - landmarks[2]) ** 2)
        dist += random.gauss(0.0, self.sense_noise)   #Ruido gaussiano en la medida
        #Z.append(dist)
	
	r = random.random()
	if r <= 0.75:
		if dist <= self.d_max:
        		return [dist, dist]
		else:
			return [-2,dist]  #Fuera de rango
	else:
		return [-1,dist]    #Medida fallida


    def gaussian(self, mu, sigma, x):
            """ calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma"""
            return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * np.pi * (sigma ** 2))

    def neff(self):
            """ Mide numero particulas que contribuyen significativamente a la distribucion de probabilidad"""
            return np.sum(np.square(self.weights))


    def update(self, Z):
        """Function to update particle weights according to the probability of observation Z."""

	self.pose_prev = copy.deepcopy(self.pose)

	if Z[0] < 0:
		pass   #Si no se recibe medida no se actualiza
	else:
		self.weights = []

		for p in self.particles:
		        
		    ## Given the particle's location 'p', which is the probability weight = prob(Z|p)
		    ## Function 'gaussian()' gives probability of a Gaussian distribution

		    d1 = sqrt((self.pose.x - p.x)**2 + (self.pose.y - p.y)**2 + (self.pose.z - p.z)**2)
		    prob = self.gaussian(Z[0], 0.5, d1)

		    self.weights.append(prob) #El peso de la particula sera su probabilidad de estar en la baliza. No se normaliza


    def resample(self):
        """Function to resample particles."""
	
	if self.neff() < self.num_particles/2:       #Solo se ejecuta el resample si menos de la mitad de las particulas tienen pesos significativos
		#print('Neff',self.neff(),'N/2',self.num_particles/2)
		#print('Resample')
		p_new = []
		index = int(random.random()*self.num_particles) #Indice aleatorio inicial
		beta = 0
		mw = max(self.weights)

		for i in range(self.num_particles):   #Recorrer particulas
		    beta += random.random() * 2 * mw    #Valor aleatorio hasta el doble del peso maximo
		    while beta > self.weights[index]:	#Si el peso es menor...
		        beta -= self.weights[index]	#Se resta peso del indice
		        index = (index + 1)%self.num_particles #Se modifica el indice
		    #Si el peso es mayor...
		    #Se introduce ruido gaussiano a la particula, y se introduce a la nueva lista de particulas
		    self.particles[index].x = self.particles[index].x + random.gauss(0.0, self.dist_noise)
		    self.particles[index].y = self.particles[index].y + random.gauss(0.0, self.dist_noise)
		    self.particles[index].z = self.particles[index].z + random.gauss(0.0, self.dist_noise)
		    p_new.append(copy.deepcopy(self.particles[index]))  #Queda una lista aleatoria de las particulas con mayor peso y algunas duplicas de estas
		    #Como los pesos no se normalizan y van en funcion de la probabilidad, no se modifican en resample

		self.particles = copy.deepcopy(p_new)
	else:
		#print('NO Resample')
		pass
		


    def publish(self,baliza):
        """Function to publish the particle cloud for visualization."""
        
	self.mediax = 0.0
	self.mediay = 0.0
	self.mediaz = 0.0
	j=0
	sx2=0.
	sy2=0.
	sz2=0.
	sxy=0.
	syz=0.
	sxz=0.
	if j==0:
		sumpesos = 1.0
	else:
		sumpesos = 0.0

        cloud_msg = PointCloud()
        cloud_msg.header.stamp = rospy.get_rostime()
        cloud_msg.header.frame_id = 'world'

        cloud_msg.points = []
        for i in range(self.num_particles):
	    j=1
            p = Point32()
            p.x = self.particles[i].x     
            p.y = self.particles[i].y
            p.z = self.particles[i].z
	    
            self.mediax += p.x * self.weights[i]  
	    self.mediay += p.y * self.weights[i]
	    self.mediaz += p.z * self.weights[i]

	    sumpesos += self.weights[i]
            cloud_msg.points.append(p)

        self.particles_publisher.publish(cloud_msg)  #Se publica la nube de puntos de las particulas

	cloud_msg = PointStamped()
        cloud_msg.header.stamp = rospy.get_rostime()
        cloud_msg.header.frame_id = 'world'

	bal = Point()
	bal.x = baliza[0]
	bal.y = baliza[1]
	bal.z = baliza[2]
	cloud_msg.point = bal
	self.balizas_publisher.publish(cloud_msg)  #Se publican la baliza del filtro

	#Media ponderada de la nube de puntos. Como los pesos no estan normalizados se dibide por el sumatorio de estos
	self.mediax = self.mediax/sumpesos
	self.mediay = self.mediay/sumpesos
	self.mediaz = self.mediaz/sumpesos

	for i in range(self.num_particles):
		#Calculo de la matriz de covarianza componente a componente
		sx2 += ((self.particles[i].x - self.mediax) ** 2)*self.weights[i]
		sy2 += ((self.particles[i].y - self.mediay) ** 2)*self.weights[i]
		sz2 += ((self.particles[i].z - self.mediaz) ** 2)*self.weights[i]
		sxy += ((self.particles[i].x - self.mediax)*(self.particles[i].y - self.mediay))*self.weights[i]
		sxz += ((self.particles[i].x - self.mediax)*(self.particles[i].z - self.mediaz))*self.weights[i]
		syz += ((self.particles[i].z - self.mediaz)*(self.particles[i].y - self.mediay))*self.weights[i]


	mean = np.array([[self.mediax, self.mediay, self.mediaz]])
	cvar = np.array([[sx2, sxy, sxz],[sxy, sy2, syz],[sxz, syz, sz2]])/sumpesos #Division por sumatorio de pesos
	#Determinante de la covarianza, medida de desviacion tipica de la media
	det = np.linalg.det(cvar)

	distp = sqrt((self.pose.x - mean[0,0]) ** 2 + (self.pose.y - mean[0,1]) ** 2 + (self.pose.z - mean[0,2]) ** 2)

	m = []
	m.append(mean)
	m.append(cvar)
	m.append(det)
	m.append(self.pose_listx)
	m.append(self.pose_listy)
	m.append(self.pose_listz)
	m.append(distp)

	return m
        

if __name__ == '__main__':
    try:
        rospy.init_node('mapping_adr', anonymous=True)

	landmarks  = [[10.5, 1.5, 0.5], [-5.0, 13.5, 4.8], [0.0, 0.5, 8.8]]   #[[0.5, 1.5, 4.5], [0.0, 0.5, 4.8], [0.0, 0.0, 4.0]] 

	fin1 = 0.0
	fin2 = 0.0
	fin3 = 0.0
	Det1 = []
	Det2 = []
	Det3 = []
	medSF1 = []
	medCF1 = []
	medSF2 = []
	medCF2 = []
	medSF3 = []
	medCF3 = []
	disp1 = []
	disp2 = []
	disp3 = []
        
	r = rospy.Rate(10) # 10hz, 100ms

        timerOn = time.time() #Temporizacion

	#Filtro de particulas por baliza
	pf1 = ParticleFilter(1,landmarks[0])
	pf2 = ParticleFilter(2,landmarks[1])
	pf3 = ParticleFilter(3,landmarks[2])

        while not rospy.is_shutdown():

		
            	#Filtro Baliza 1
            	#if pf1.check_robot_motion():
                if fin1 == 0:
		     	# Create sensor measurement
		  	Z = pf1.sense(landmarks[0])
			medSF1.append(Z[1])
			medCF1.append(Z[0])

			# Particle Filter steps
			if Z[0]>=0:
				pf1.update(Z)
				pf1.resample()
			    
		    	# Publish particles
		    	M = pf1.publish(landmarks[0])
			Det1.append(M[2])			
			disp1.append(M[6])

			#Condicion de convergencia
			if M[2] < 0.02 and M[2] > 0.:
		  		print('Z1',Z)
				print('B1',landmarks[0])
				print('Media1',M[0])
				print('CVar1',M[1])
				print('Det1',M[2])
				fin1 = 1.0

		"""FIN F1"""

            	#Filtro Baliza 2
            	#if pf2.check_robot_motion():
                if fin2 == 0:
		       	# Create sensor measurement
		      	Z = pf2.sense(landmarks[1])
			medSF2.append(Z[1])
			medCF2.append(Z[0])

			# Particle Filter steps
			if Z[0]>=0:
				pf2.update(Z)
				pf2.resample()
		    
		    	# Publish particles
		    	M = pf2.publish(landmarks[1])
			Det2.append(M[2])			
			disp2.append(M[6])			

			#Condicion de convergencia
			if M[2] < 0.02 and M[2] > 0.:
		    		print('Z2',Z)
				print('B2',landmarks[1])
				print('Media2',M[0])
				print('CVar2',M[1])
				print('Det2',M[2])
				fin2 = 1.0

		"""FIN F2"""

            	#Filtro Baliza 3
            	#if pf3.check_robot_motion():
                if fin3 == 0:
		       	# Create sensor measurement
		       	Z = pf3.sense(landmarks[2])
			medSF3.append(Z[1])
			medCF3.append(Z[0])

			# Particle Filter steps
			if Z[0]>=0:
				pf3.update(Z)
				pf3.resample()	
		    
		    	# Publish particles
		    	M = pf3.publish(landmarks[2])
			Det3.append(M[2])			
			disp3.append(M[6])			

			#Condicion de convergencia
			if M[2] < 0.02 and M[2] > 0.:
		    		print('Z3',Z)
				print('B3',landmarks[2])
				print('Media3',M[0])
				print('CVar3',M[1])
				print('Det3',M[2])
				fin3 = 1.0

		"""FIN F3"""

		#Representacion graficas 
		if fin1 == 1 and fin2 == 1 and fin3 == 1:

			#Tiempo de simulacion
			timerOff = time.time()
			#print('TimeOff',timerOff)
			T = timerOff-timerOn
			print('Time',T)


			#Graficas de trayectoria en el espacio 
			fig = plt.figure()
			ax = Axes3D(fig)
			zline = M[5]
			xline = M[3]
			yline = M[4]
			ax.plot3D(xline,yline,zline,'gray') #3D
			ax.set_xlabel("X (m)")
			ax.set_ylabel("Y (m)")
			ax.set_zlabel("Z (m)")
			ax.legend(["Trayectoria"])
			plt.suptitle("Trayectoria 3D")
			plt.grid()
			plt.show()

			Tamx = len(xline)
			Paso = T/Tamx
			time = np.zeros(Tamx)
			suma = 0.
			for i in range(len(time)):
				suma += Paso
				time[i] = suma
			
			#Graficas de trayectoria en el plano
			fig, ax = plt.subplots(3,1,sharey = True)
			ax[0].plot(time,xline) #2D, x
			ax[0].set_ylabel("X (m)")
			ax[0].grid()
			ax[0].legend(["Trayectoria X"])
			ax[1].plot(time,yline) #2D, y
			ax[1].set_ylabel("Y (m)")
			ax[1].grid()
			ax[1].legend(["Trayectoria Y"])
			ax[2].plot(time,zline) #2D, z
			ax[2].set_xlabel("Tiempo (s)")
			ax[2].set_ylabel("Z (m)")
			ax[2].grid()
			ax[2].legend(["Trayectoria Z"])
			plt.suptitle("Evolucion de las componentes XYZ")
			plt.show()
			

			while len(Det1) < len(time):
				Det1.append(Det1[len(Det1)-1])
			while len(Det2) < len(time):
				Det2.append(Det2[len(Det2)-1])
			while len(Det3) < len(time):
				Det3.append(Det3[len(Det3)-1])

			#Evolucion de determinantes
			fig, ax = plt.subplots(3,1,sharey = True)
			ax[0].plot(time,Det1) 
			ax[0].set_ylabel("Det1 (m)")
			ax[0].grid()
			ax[0].legend(["Det 1"])
			ax[1].plot(time,Det2) 
			ax[1].set_ylabel("Det2 (m)")
			ax[1].grid()
			ax[1].legend(["Det 2"])
			ax[2].plot(time,Det3) 
			ax[2].set_xlabel("Tiempo (s)")
			ax[2].set_ylabel("Det3 (m)")
			ax[2].grid()
			ax[2].legend(["Det 3"])
			plt.suptitle("Evolucion de los determinantes")
			plt.show()

			while len(medSF1) < len(time):
				medSF1.append(medSF1[len(medSF1)-1])
				medCF1.append(medCF1[len(medCF1)-1])
				disp1.append(disp1[len(disp1)-1])
			while len(medSF2) < len(time):
				medSF2.append(medSF2[len(medSF2)-1])
				medCF2.append(medCF2[len(medCF2)-1])
				disp2.append(disp2[len(disp2)-1])
			while len(medSF3) < len(time):
				medSF3.append(medSF3[len(medSF3)-1])
				medCF3.append(medCF3[len(medCF3)-1])
				disp3.append(disp3[len(disp3)-1])

			dm1 = np.array(medSF1)
			dm2 = np.array(medSF2)
			dm3 = np.array(medSF3)
			dp1 = np.array(disp1)
			dp2 = np.array(disp2)
			dp3 = np.array(disp3)
			

			#Medidas recibidas
			fig, ax = plt.subplots(3,1,sharey = True)
			ax[0].plot(time,medCF1) 
			ax[0].set_ylabel("Medida B1 (m)")
			ax[0].grid()
			ax[0].legend(["Distancia B1"])
			ax[1].plot(time,medCF2) 
			ax[1].set_ylabel("Medida B2 (m)")
			ax[1].grid()
			ax[1].legend(["Distancia B2"])
			ax[2].plot(time,medCF3) 
			ax[2].set_xlabel("Tiempo (s)")
			ax[2].set_ylabel("Medida B3 (m)")
			ax[2].grid()
			ax[2].legend(["Distancia B3"])
			plt.suptitle("Medidas de distancias recibidas")
			plt.show()

			#Distancia baliza y nube de puntos
			fig, ax = plt.subplots(3,1,sharey = True) 
			ax[0].plot(time,medSF1) 
			ax[0].plot(time,disp1) 
			ax[0].set_ylabel("Distancia (m)")
			ax[0].grid()
			ax[0].legend(["Distancia B1", "Distancia P1"])
			ax[1].plot(time,medSF2) 
			ax[1].plot(time,disp2)
			ax[1].set_ylabel("Distancia (m)")
			ax[1].grid()
			ax[1].legend(["Distancia B2", "Distancia P2"])
			ax[2].plot(time,medSF3)
			ax[2].plot(time,disp3) 
			ax[2].set_xlabel("Tiempo (s)")
			ax[2].set_ylabel("Distancia (m)")
			ax[2].grid()
			ax[2].legend(["Distancia B3", "Distancia P3"])
			plt.suptitle("Distancias a balizas y a medias de particulas")
			plt.show()

			#Error de media de nube de puntos 
			fig, ax = plt.subplots(3,1,sharey = True) 
			ax[0].plot(time,abs(dm1-dp1)) 
			ax[0].set_ylabel("Error (m)")
			ax[0].grid()
			ax[0].legend(["Error 1"])
			ax[1].plot(time,abs(dm2-dp2))
			ax[1].set_ylabel("Error (m)")
			ax[1].grid()
			ax[1].legend(["Error 2"])
			ax[2].plot(time,abs(dm3-dp3))
			ax[2].set_xlabel("Tiempo (s)")
			ax[2].set_ylabel("Error (m)")
			ax[2].grid()
			ax[2].legend(["Error 3"])
			plt.suptitle("Errores entre las nubes de puntos y las balizas")
			plt.show()

			fin1 = 2
			fin2 = 2
			fin3 = 2

        	r.sleep()

    except rospy.ROSInterruptException:
        pass
