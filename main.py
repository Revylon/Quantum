from tkinter import *
from tkinter import messagebox as mb
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, integrate
from scipy.optimize import fsolve
import astropy.units as u
import astropy.constants as c
import mpmath as mp


# Константы 

NUM_LAYERS = 1
VECTOR_D = []
VECTOR_W = []
VECTOR_MASS = []
VECTOR_EXP = []
TOLL = 10
ENERGY_ROOTS = []
ME = (c.m_e.to(u.g)).value
H = c.hbar.to(u.erg*u.s).value # Значение постоянной планка (СГС)


class EnergyFinding():
	""" Класс нахождения уровней энергии. Все значения необходимо вводить в СГС.
		Энергии в Электронвольтах!!!  """

	def __init__(self, number_of_layers, vector_d, vector_W, mass_vector):
		''' Инициализация входных данных'''

		# Колличество уровней с постоянной эффективной массой 
		#    в гетероструктуре
		self.number_of_layers = number_of_layers
		# Вектор размерности n с размерами каждого слоя
		self.structure_width_vector = vector_d
		# Вектор размерности n с потенциалом для каждого уровня
		self.barriers_high_vector = vector_W
		# Вектор размерности n для эффективных масс в каждом слое
		self.mass_vector = mass_vector

		self.roots = []

	def matrix_M(self, ind_layer, Energy):
		# Функция создающая матрицу перехода через гетерограницу
		E = Energy
		mass = self.mass_vector[ind_layer]
		barriers_high = self.barriers_high_vector[ind_layer]

		k = mp.sqrt(2*mass*(E-barriers_high)/H**2)
		M = mp.matrix(2)
		M[0,0] = 1
		M[0,1] = 1
		M[1,0] = 1j*k/mass
		M[1,1] = -1j*k/mass

		return M


	def matrix_invers(self, Matrix):
		# Функция создающая обратную матрицу 

		# Определитель матрицы
		matrix_det = Matrix[0,0]*Matrix[1,1] - Matrix[0,1]*Matrix[1,0]
		
		M = mp.matrix(2)
		M[0,0] = Matrix[1,1]/matrix_det
		M[0,1] = -Matrix[0,1]/matrix_det
		M[1,0] = -Matrix[1,0]/matrix_det
		M[1,1] = Matrix[0,0]/matrix_det

		return M

	def matrix_N(self, ind_layer, Energy):
		# Функция создающая матрицу переноса через один слой 

		E = Energy
		mass = self.mass_vector[ind_layer]
		barriers_high = self.barriers_high_vector[ind_layer]
		layer_width = self.structure_width_vector[ind_layer]
		k = mp.sqrt(2*mass*(E-barriers_high)/H**2)
		N = mp.matrix(2)
		N[0,0] = mp.exp(1j*k*layer_width)
		N[0,1] = 0
		N[1,0] = 0
		N[1,1] = mp.exp(-1j*k*layer_width)

		return N

	def full_transfer_matrix(self, Energy):
		# Функция, создающая полную матрицу переноса через всю заданную гетероструктуру
		M = []
		Mt = []
		N = []
		T = mp.eye(2)

		for number in range(self.number_of_layers):
			M.append(self.matrix_M(number,Energy))
			Mt.append(self.matrix_invers(self.matrix_M(number,Energy)))
			

			if not number == min(range(self.number_of_layers)) and not number== max(range(self.number_of_layers)):
				N.append(self.matrix_N(number,Energy))
			else:
				N.append(0)

	
		# Полная матрица переноса через всю структуру
		for number in range(self.number_of_layers - 1):
			if number == max(range(self.number_of_layers - 1)):
				T = Mt[number+1]*M[number]*T
			else:
				T = N[number+1]*Mt[number+1]*M[number]*T

		# Возвращает элемент Т(2,2) матрицы Т
		return T[1,1]


	def find_roots(self):
		# Вектор E0 - вектор начального приближения
		E_min = 0.001*max(self.barriers_high_vector) #+ 0.001*10**(-12)
		E_max = 0.999*max(self.barriers_high_vector) #- 0.001*10**(-12)

		E_list = np.linspace(E_min,E_max,100)
		T_list = np.zeros_like(E_list)

		for energy in range(len(E_list)):
			T_list[energy] = float((self.full_transfer_matrix(E_list[energy]).real))

		sign_vector = []
		zero_vector = []
		
		
		for energy in range(len(E_list)):
			if T_list[energy] > 0:
				sign_vector.append(1)
			elif T_list[energy] <0:
				sign_vector.append(-1)

		for ind in range(len(sign_vector)-1):
			if not sign_vector[ind] == sign_vector[ind+1]:
				zero_vector.append(E_list[ind])

		for energy in range(len(zero_vector)):
			self.roots.append(float(mp.findroot(lambda t: self.full_transfer_matrix(t).real, zero_vector[energy], solver = 'mnewton')).real)

		return self.roots


	def en_plot(self):
		# Функция вывода графика зависимости Т(2,2) от Энергии
		E_min = 0.001*max(self.barriers_high_vector)
		E_max = 0.999*max(self.barriers_high_vector)

		E_list = np.linspace(E_min,E_max,100)
		T_list = np.zeros_like(E_list)

		for energy in range(len(E_list)):
			T_list[energy] = float((self.full_transfer_matrix(E_list[energy]).real))

		plt.plot(E_list/10**(-13),T_list)
		plt.grid()
		for ind in range(len(self.roots)):
			plt.axvline(x=self.roots[ind]/10**(-13), ls='--', color='r')
		plt.xlabel(r'$E,10^{-13} Эрг$')
		plt.ylabel(r'$T_{11}(E)$')
		plt.show()


class VaweFunctionFinding():
	""" Класс нахождения волновых функций, 
	соответствующих определенному уровню энергии"""
	def __init__(self, number_of_layers, vector_d,vector_W, mass_vector, roots_vector):
		# Колличество слоёв в структуре
		self.number_of_layers = number_of_layers
		# Вектор ширин слоёв структуры
		self.structure_width_vector = vector_d
		# Вектор размерности n с потенциалом для каждого уровня
		self.barriers_high_vector = vector_W
		# Вектор размерности n для эффективных масс в каждом слое
		self.mass_vector = mass_vector
		# Вектор граничных значений для вычисления в.ф. на грфницах
		self.x_vect = np.zeros_like(self.structure_width_vector)

		self.x_max = 0
		for ind in range(len(self.structure_width_vector)):
			if not ind ==0:
				self.x_max = self.x_max + self.structure_width_vector[ind]

		# Вектор корней энергий
		self.roots = roots_vector

		for ind in range(self.number_of_layers):
			if ind == 0 or ind == 1:
				continue
			else:
				self.x_vect[ind] = self.structure_width_vector[(ind-1)]

		for ind in range(self.number_of_layers):
			if ind == 0:
				continue
			else:
				self.x_vect[ind] = self.x_vect[ind-1] + self.x_vect[ind]

	def matrix_M(self, ind_layer, Energy):
		# Функция создающая матрицу перехода через гетерограницу
		E = Energy
		mass = self.mass_vector[ind_layer]
		barriers_high = self.barriers_high_vector[ind_layer]

		k = mp.sqrt(2*mass*(E-barriers_high)/H**2)
		M = mp.matrix(2)
		M[0,0] = 1
		M[0,1] = 1
		M[1,0] = 1j*k/mass
		M[1,1] = -1j*k/mass

		return M


	def matrix_invers(self, Matrix):
		# Функция создающая обратную матрицу 

		# Определитель матрицы
		matrix_det = Matrix[0,0]*Matrix[1,1] - Matrix[0,1]*Matrix[1,0]
		
		M = mp.matrix(2)
		M[0,0] = Matrix[1,1]/matrix_det
		M[0,1] = -Matrix[0,1]/matrix_det
		M[1,0] = -Matrix[1,0]/matrix_det
		M[1,1] = Matrix[0,0]/matrix_det

		return M



	def matrix_N(self, ind_layer, Energy,x):
		
		E = Energy
		mass = self.mass_vector[ind_layer]
		barriers_high = self.barriers_high_vector[ind_layer]
		layer_width = self.structure_width_vector[ind_layer]
		
		k = mp.sqrt(2*mass*(E-barriers_high)/H**2)



		return mp.matrix([[mp.exp(1j*k*(x - self.x_vect[ind_layer])),0],[0,mp.exp(-1j*k*(x - self.x_vect[ind_layer]))]])




	def matrix_C(self,ind_layer,Energy,x):

		C_vector = mp.matrix([[0],[1]])
		E = Energy

		if ind_layer == 0:
			return self.matrix_N(0,E,x)*C_vector

		else:
			return self.matrix_N(ind_layer,E,x)*self.matrix_invers(self.matrix_M(ind_layer,E))*self.matrix_M(ind_layer-1,E)*self.matrix_C(ind_layer-1,E,self.x_vect[ind_layer])




	def vawe_function(self,x, Energy):

		
		Vawe_vector = []

		for index in range(self.number_of_layers):
			Vawe_vector.append((self.matrix_C(index,Energy,x)[0,0] + self.matrix_C(index,Energy,x)[1,0]).real)


		for ind in range(self.number_of_layers):
			if ind == min(range(self.number_of_layers)):
				if x<= self.x_vect[ind] and x>= self.structure_width_vector[ind]:
					return Vawe_vector[ind]

			elif ind == max(range(self.number_of_layers)):
				if x>= self.x_vect[ind]: #and x <= self.structure_width_vector[ind]:
					return Vawe_vector[ind]

			else: #elif ind > min(range(self.number_of_layers)) and ind < max(range(self.number_of_layers)):
				if x >= self.x_vect[ind] and x <= self.x_vect[ind+1]:
					return Vawe_vector[ind]

	def U(self,x):
		a = np.zeros(len(self.structure_width_vector)+1)

		for num in range(len(a)):
			if num ==0:
				a[num] = self.structure_width_vector[num]
			elif num ==1:
				a[num] = 0
			else:
				a[num] = self.structure_width_vector[num-1]
				a[num] = a[num]+a[num-1]

		for num in range(len(self.structure_width_vector)):
			if x>=a[num] and x<= a[num+1]:
				return (self.barriers_high_vector[num]*u.erg).to(u.eV).value*10





	def vawe_func_plot(self, Energy):

		x_min = min(self.structure_width_vector)
		#x_max = max(self.structure_width_vector)
		x = np.linspace(x_min,self.x_max,150)
		U = np.zeros_like(x)

		for ind in range(len(x)):
			U[ind] = self.U(x[ind])

		v_F_vector = mp.matrix(len(Energy),len(x))
		

		for energy in range(len(Energy)):
			for ind in range(len(x)):
				v_F_vector[energy,ind] = self.vawe_function(x[ind],Energy[energy])


		plt.plot(x/10**(-7), U, color='k',linestyle = '--', linewidth = 1)
		for energy in range(len(Energy)):
			plt.plot(x/10**(-7) , v_F_vector[energy,:],linewidth = 2)
			
		
		plt.ylabel(r'$\Psi_{k}(x),  U(x)$')
		plt.xlabel(r'$x,10^{-9} м$')

		plt.show()





	def vawe_layer(self, ind_layer_energy):

		x_min = min(self.structure_width_vector)
		#x_max = max(self.structure_width_vector)
		x_1 = np.linspace(x_min,self.x_max,200)
		v_F_vector = np.zeros_like(x_1)

		for ind in range(len(x_1)):
			v_F_vector[ind] = self.vawe_function(x_1[ind],self.roots[ind_layer_energy])

		return v_F_vector




	def norm_vawe_function(self, ind_layer_energy):

		C_norm = (mp.sqrt(mp.quad(lambda t:(self.vawe_function(t, self.roots[ind_layer_energy]))**2, [min(self.structure_width_vector),self.x_max])).real)
		#print(C_norm)
		return self.vawe_layer(ind_layer_energy)/C_norm




	def norm_funct_plot(self,ind_layer_energy):

		vawe = self.norm_vawe_function(ind_layer_energy)
		x = np.linspace(min(self.structure_width_vector), self.x_max,200)
		plt.plot(x/10**(-9),vawe)
		plt.show()


class EnergyFinding_exp():
	""" Класс нахождения уровней энергии. Все значения необходимо вводить в СГС.
		Энергии в Электронвольтах!!!  """

	def __init__(self, number_of_layers, vector_d, vector_W, mass_vector,exp_index,toll):
		''' Инициализация входных данных'''

		# Колличество уровней с постоянной эффективной массой 
		#    в гетероструктуре
		self.number_of_layers = number_of_layers
		# Вектор размерности n с размерами каждого слоя
		self.structure_width_vector = vector_d
		# Вектор размерности n с потенциалом для каждого уровня
		self.barriers_high_vector = vector_W
		# Вектор размерности n для эффективных масс в каждом слое
		self.mass_vector = mass_vector
		self.toll = toll

		self.exp_index_list = exp_index
		
		self.roots = []

		# Разбиение экспонециального барьера на ступеньки
		new_D = np.zeros(self.toll*len(self.exp_index_list)+(self.number_of_layers-len(self.exp_index_list)))
		new_W = np.zeros(self.toll*len(self.exp_index_list)+(self.number_of_layers-len(self.exp_index_list))) 
		
		k=0
		if not self.exp_index_list == []:
			for i in range(self.number_of_layers):
				if i not in self.exp_index_list:
					new_D[i+k] = self.structure_width_vector[i]
					new_W[i+k] = self.barriers_high_vector[i]
				else:
					for n in range(self.toll):
						new_D[i+k+n] = self.structure_width_vector[i]/self.toll
						new_W[i+k+n] = self.barriers_high_vector[i]*(1-mp.exp(-(10*n/self.toll)))
					k = k+(self.toll-1)
		else:
			new_D = self.structure_width_vector
			new_W = self.barriers_high_vector

		self.structure_width_vector = new_D
		self.barriers_high_vector = new_W


		self.x_max = 0
		for ind in range(len(self.structure_width_vector)):
			if not ind ==0:
				self.x_max = self.x_max + self.structure_width_vector[ind]



		self.a = np.zeros(len(self.structure_width_vector)+1)
		for num in range(len(self.a)):
			if num ==0:
				self.a[num] = self.structure_width_vector[num]
			elif num ==1:
				self.a[num] = 0
			else:
				self.a[num] = self.structure_width_vector[num-1]
				self.a[num] = self.a[num] + self.a[num-1]





	def U(self,x):

		for num in range(len(self.structure_width_vector)):
			if x>= self.a[num] and x<= self.a[num+1]:
				return (self.barriers_high_vector[num])


	def matrix_M(self, ind_layer, Energy,x):
		# Функция создающая матрицу перехода через гетерограницу

		E = Energy
		mass = self.mass_vector[ind_layer]

		k = mp.sqrt(2*mass*(E-self.U(x))/H**2)
		M = mp.matrix(2)
		M[0,0] = 1
		M[0,1] = 1
		M[1,0] = 1j*k/mass
		M[1,1] = -1j*k/mass

		return M


	def matrix_invers(self, Matrix):
		# Функция создающая обратную матрицу 

		# Определитель матрицы
		matrix_det = Matrix[0,0]*Matrix[1,1] - Matrix[0,1]*Matrix[1,0]
		
		M = mp.matrix(2)
		M[0,0] = Matrix[1,1]/matrix_det
		M[0,1] = -Matrix[0,1]/matrix_det
		M[1,0] = -Matrix[1,0]/matrix_det
		M[1,1] = Matrix[0,0]/matrix_det

		return M

	def matrix_N(self, ind_layer, Energy,x):
		# Функция создающая матрицу переноса через один слой 

		E = Energy
		mass = self.mass_vector[ind_layer]
		barriers_high = self.U(x)
		layer_width = self.structure_width_vector[ind_layer]
		k = mp.sqrt(2*mass*(E-barriers_high)/H**2)
		N = mp.matrix(2)
		N[0,0] = mp.exp(1j*k*layer_width)
		N[0,1] = 0
		N[1,0] = 0
		N[1,1] = mp.exp(-1j*k*layer_width)

		return N

	def full_transfer_matrix(self, Energy):
		# Функция, создающая полную матрицу переноса через всю заданную гетероструктуру
		M = []
		Mt = []
		N = []
		E = Energy
		T = mp.eye(2)

		a = np.zeros(len(self.structure_width_vector)+1)

		# Полная матрица переноса через всю структуру
		for number in range(self.number_of_layers - 1):
			if number == max(range(self.number_of_layers - 1)):
				T = self.matrix_invers(self.matrix_M(number+1,E,self.a[number+2]))*self.matrix_M(number,E,self.a[number+1])*T

			else:
				T = self.matrix_N(number+1,E,self.a[number+2])*self.matrix_invers(self.matrix_M(number+1,E,self.a[number+2]))*self.matrix_M(number,E,self.a[number+1])*T

		# Возвращает элемент Т(2,2) матрицы Т
		return T[1,1]


	def find_roots_exp(self):

		x_min = min(self.structure_width_vector)
		x = np.linspace(x_min,self.x_max,100)
		# Вектор E0 - вектор начального приближения
		E_min = 0.001*max(self.barriers_high_vector)
		E_max = 0.999*max(self.barriers_high_vector)

		E_list = np.linspace(E_min,E_max,100)
		T_list = np.zeros_like(E_list)

		for energy in range(len(E_list)):
			for i in range(len(x)):
				T_list[energy] = float((self.full_transfer_matrix(E_list[energy]).real))

		sign_vector = []
		zero_vector = []
		
		
		for energy in range(len(E_list)):
			if T_list[energy] > 0:
				sign_vector.append(1)
			elif T_list[energy] <0:
				sign_vector.append(-1)

		for ind in range(len(sign_vector)-1):
			if not sign_vector[ind] == sign_vector[ind+1]:
				zero_vector.append(E_list[ind])

		for energy in range(len(zero_vector)):
			self.roots.append(float(mp.findroot(lambda t: self.full_transfer_matrix(t).real, zero_vector[energy], solver = 'mnewton')).real)

		return self.roots


	def en_plot_exp(self):
		# Функция вывода графика зависимости Т(2,2) от Энергии
		x_min = min(self.structure_width_vector)
		x = np.linspace(x_min,self.x_max,100)

		E_min = 0.001*max(self.barriers_high_vector)
		E_max = 0.999*max(self.barriers_high_vector)

		E_list = np.linspace(E_min,E_max,100)
		T_list = np.zeros_like(E_list)


		for energy in range(len(E_list)):
			T_list[energy] = float((self.full_transfer_matrix(E_list[energy]).real))

		plt.plot(E_list/10**(-13),T_list)
		plt.grid()
		for ind in range(len(self.roots)):
			plt.axvline(x=self.roots[ind]/10**(-13), ls='--', color='r')
		plt.xlabel(r'$E,10^{-13} Эрг$')
		plt.ylabel(r'$T_{11}(E)$')
		plt.show()

	def en_plot_U(self):

		x_min = min(self.structure_width_vector)
		#x_max = max(self.structure_width_vector)
		x = np.linspace(x_min,self.x_max,1000)
		U_p = np.zeros_like(x)
			
		for ind in range(len(x)):
			U_p[ind] = ((self.U(x[ind]))*u.erg).to(u.eV).value

		plt.plot(x/10**(-7), U_p, color='r',linestyle = '--', linewidth = 1)
		plt.ylabel(r'$U(x)$')
		plt.xlabel(r'$x,10^{-7} cм$')
		plt.show()


class VaweFunctionFinding_exp():
	""" Класс нахождения волновых функций, 
	соответствующих определенному уровню энергии"""
	def __init__(self, number_of_layers, vector_d,vector_W, mass_vector, roots_vector,exp_index,toll):
		''' Инициализация входных данных'''
		
		# Вектор корней энергий
		self.roots = roots_vector
		# Колличество уровней с постоянной эффективной массой 
		#    в гетероструктуре
		self.number_of_layers = number_of_layers
		# Вектор размерности n с размерами каждого слоя
		self.structure_width_vector = vector_d
		# Вектор размерности n с потенциалом для каждого уровня
		self.barriers_high_vector = vector_W
		# Вектор размерности n для эффективных масс в каждом слое
		self.mass_vector = mass_vector
		# Массив из индексов экспоненциальных барьеров
		self.exp_index_list = exp_index

		self.toll = toll
		
		# Разбиение экспонециального барьера на ступеньки
		new_D = np.zeros(self.toll*len(self.exp_index_list)+(self.number_of_layers-len(self.exp_index_list)))
		new_W = np.zeros(self.toll*len(self.exp_index_list)+(self.number_of_layers-len(self.exp_index_list))) 
		
		k=0
		if not self.exp_index_list == []:
			for i in range(self.number_of_layers):
				if i not in self.exp_index_list:
					new_D[i+k] = self.structure_width_vector[i]
					new_W[i+k] = self.barriers_high_vector[i]
				else:
					for n in range(self.toll):
						new_D[i+k+n] = self.structure_width_vector[i]/self.toll
						new_W[i+k+n] = self.barriers_high_vector[i]*(1-mp.exp(-(10*n/self.toll)))
					k = k+(self.toll-1)
		else:
			new_D = self.structure_width_vector
			new_W = self.barriers_high_vector

		self.structure_width_vector = new_D
		self.barriers_high_vector = new_W

		self.x_max = 0
		for ind in range(len(self.structure_width_vector)):
			if not ind ==0:
				self.x_max = self.x_max + self.structure_width_vector[ind]

		self.a = np.zeros(len(self.structure_width_vector)+1)
		for num in range(len(self.a)):
			if num ==0:
				self.a[num] = self.structure_width_vector[num]
			elif num ==1:
				self.a[num] = 0
			else:
				self.a[num] = self.structure_width_vector[num-1]
				self.a[num] = self.a[num] + self.a[num-1]

		# Вектор граничных значений для вычисления в.ф. на грфницах
		self.x_max = 0
		for ind in range(len(self.structure_width_vector)):
			if not ind ==0:
				self.x_max = self.x_max + self.structure_width_vector[ind]

		self.x_vect = np.zeros_like(self.structure_width_vector)
		for ind in range(self.number_of_layers):
			if ind == 0 or ind == 1:
				continue
			else:
				self.x_vect[ind] = self.structure_width_vector[(ind-1)]

		for ind in range(self.number_of_layers):
			if ind == 0:
				continue
			else:
				self.x_vect[ind] = self.x_vect[ind-1] + self.x_vect[ind]

	def U(self,x):
		
		for num in range(len(self.structure_width_vector)):
			if x>= self.a[num] and x<= self.a[num+1]:
				return (self.barriers_high_vector[num])


	def matrix_M(self, ind_layer, Energy):
		# Функция создающая матрицу перехода через гетерограницу

		E = Energy
		mass = self.mass_vector[ind_layer]
		barriers_high = self.barriers_high_vector[ind_layer]

		k = mp.sqrt(2*mass*(E-barriers_high)/H**2)
		M = mp.matrix(2)
		M[0,0] = 1
		M[0,1] = 1
		M[1,0] = 1j*k/mass
		M[1,1] = -1j*k/mass

		return M


	def matrix_invers(self, Matrix):
		# Функция создающая обратную матрицу 

		# Определитель матрицы
		matrix_det = Matrix[0,0]*Matrix[1,1] - Matrix[0,1]*Matrix[1,0]
		
		M = mp.matrix(2)
		M[0,0] = Matrix[1,1]/matrix_det
		M[0,1] = -Matrix[0,1]/matrix_det
		M[1,0] = -Matrix[1,0]/matrix_det
		M[1,1] = Matrix[0,0]/matrix_det

		return M



	def matrix_N(self, ind_layer, Energy,x):
		
		E = Energy
		mass = self.mass_vector[ind_layer]
		barriers_high = self.barriers_high_vector[ind_layer]
		layer_width = self.structure_width_vector[ind_layer]
		
		k = mp.sqrt(2*mass*(E-barriers_high)/H**2)

		return mp.matrix([[mp.exp(1j*k*(x - self.x_vect[ind_layer])),0],[0,mp.exp(-1j*k*(x - self.x_vect[ind_layer]))]])




	def matrix_C(self,ind_layer,Energy,x):

		C_vector = mp.matrix([[0],[1]])
		E = Energy

		if ind_layer == 0:
			return self.matrix_N(0,E,x)*C_vector

		else:
			return self.matrix_N(ind_layer,E,x)*self.matrix_invers(self.matrix_M(ind_layer,E))*self.matrix_M(ind_layer-1,E)*self.matrix_C(ind_layer-1,E,self.x_vect[ind_layer])



	def vawe_function(self,x, Energy):

		
		Vawe_vector = []

		for index in range(self.number_of_layers):
			Vawe_vector.append((self.matrix_C(index,Energy,x)[0,0] + self.matrix_C(index,Energy,x)[1,0]).real)


		for ind in range(self.number_of_layers):
			if ind == min(range(self.number_of_layers)):
				if x<= self.x_vect[ind] and x>= self.structure_width_vector[ind]:
					return Vawe_vector[ind]

			elif ind == max(range(self.number_of_layers)):
				if x>= self.x_vect[ind]:
					return Vawe_vector[ind]

			else: 
				if x >= self.x_vect[ind] and x <= self.x_vect[ind+1]:
					return Vawe_vector[ind]




	def vawe_func_plot_exp(self, Energy):

		x_min = min(self.structure_width_vector)
		x = np.linspace(x_min,self.x_max,150)
		U = np.zeros_like(x)

		for ind in range(len(x)):
			U[ind] = ((self.U(x[ind]))*u.erg).to(u.eV).value

		v_F_vector = mp.matrix(len(Energy),len(x))
		

		for energy in range(len(Energy)):
			for ind in range(len(x)):
				v_F_vector[energy,ind] = self.vawe_function(x[ind],Energy[energy])


		plt.plot(x/10**(-7), U, color='k',linestyle = '--', linewidth = 1)
		for energy in range(len(Energy)):
			plt.plot(x/10**(-7) , v_F_vector[energy,:]*0.1,linewidth = 2)
			
		
		plt.ylabel(r'$\Psi_{k}(x),  U(x)$')
		plt.xlabel(r'$x,10^{-7} cм$')

		plt.show()



	def vawe_layer(self, ind_layer_energy):

		x_min = min(self.structure_width_vector)
		#x_max = max(self.structure_width_vector)
		x_1 = np.linspace(x_min,self.x_max,200)
		v_F_vector = np.zeros_like(x_1)

		for ind in range(len(x_1)):
			v_F_vector[ind] = self.vawe_function(x_1[ind],self.roots[ind_layer_energy])

		return v_F_vector



	def norm_vawe_function(self, ind_layer_energy):

		C_norm = (mp.sqrt(mp.quad(lambda t:(self.vawe_function(t, self.roots[ind_layer_energy]))**2, [min(self.structure_width_vector),self.x_max])).real)
		#print(C_norm)
		return self.vawe_layer(ind_layer_energy)/C_norm




	def norm_funct_plot_exp(self,ind_layer_energy):

		vawe = self.norm_vawe_function(ind_layer_energy)
		x = np.linspace(min(self.structure_width_vector), self.x_max,200)
		plt.plot(x/10**(-9),vawe)
		plt.show()


class Main():
	def __init__(self,master):
		
		self.master = master
		self.master.title("Geterostructure Solver")
		self.master.geometry('300x150+100+200')

		self.btn1 = Button(self.master,text="Linear",width=25,height=3, command=self.linear_solver).grid(row=0,column=0)
		self.btn2 = Button(self.master,text="Exponential",width=25,height=3, command=self.exponential_solver).grid(row=1,column=0)


	def linear_solver(self):
		root2 = Toplevel(self.master)
		myGUI = Linear(root2)

	def exponential_solver(self):
		root3 = Toplevel(self.master)
		myGUI = Exponential(root3)


class Linear():
	def __init__(self,master):
		
		self.master = master
		self.master.geometry("800x600+200+200")
		self.master.title("Linear_Solver")

		self.NUM_LAYERS = 1
		self.VECTOR_D = []
		self.VECTOR_W = []
		self.VECTOR_MASS = []
		self.ENERGY_ROOTS = []
		self.ME = (c.m_e.to(u.g)).value

		self.label_num_layers = Label(self.master, width=20, font ='arial 18')
		self.label_num_layers['text'] = "Количество слоёв"
		self.label_vector_d = Label(self.master, width=20, font ='arial 18')
		self.label_vector_d['text'] = "Толщина слоёв, нм"
		self.label_vector_W = Label(self.master, width=20, font ='arial 18')
		self.label_vector_W['text'] = "Потенциал слоёв, эВ"
		self.label_vector_mass = Label(self.master,width=20, font ='arial 18')
		self.label_vector_mass['text'] = "Массы слоёв, m/m0"
		
 
		self.entry_num_layers = Entry(self.master, width=20, font=15)
		self.entry_vector_d = Entry(self.master, width=20, font=15)
		self.entry_vector_W = Entry(self.master, width=20, font=15)
		self.entry_vector_mass = Entry(self.master, width=20, font=15)
		self.entry_norm_index = Entry(self.master, width=20, font=15)

		self.but_Energy = Button(self.master,font=('Ubuntu',12),width=25,height=3, text='Рассчитать уровни энергии')
		self.but_VaweFunction = Button(self.master,font=('Ubuntu',12),width=25,height=3, text='Рассчитать волновые функции')
		self.but_Select = Button(self.master,font=('Ubuntu',12),width=25,height=3, text='Принять')
		self.but_One_Function_norm = Button(self.master,font=('Ubuntu',12),width=25,height=3, text='Нормированная в.ф.')
		
		self.txt = Text(self.master,height=5,width=30)

		self.label_num_layers.grid(row=0, column=0, pady = 5)
		self.label_vector_d.grid(row=1, column=0,pady = 5)
		self.label_vector_W.grid(row=2, column=0,pady = 5)
		self.label_vector_mass.grid(row=3, column=0,pady = 5)

		self.entry_num_layers.grid(row=0, column=1, ipady = 3)
		self.entry_vector_d.grid(row=1, column=1, ipady = 3)
		self.entry_vector_W.grid(row=2, column=1, ipady = 3)
		self.entry_vector_mass.grid(row=3, column=1, ipady = 3)

		self.but_Select.grid(row=9, column=0)
		self.but_Energy.grid(row=10, column=0, pady = 35)
		self.but_VaweFunction.grid(row=11, column=0)
		self.txt.grid(row=10,column=1)
		self.but_Energy.bind("<Button-1>",self.find_roots)
		self.but_VaweFunction.bind("<Button-1>", self.find_vawe_funct)
		self.but_Select.bind("<Button-1>",self.set_values)


	def set_values(self,event):
		
		self.NUM_LAYERS = int(self.entry_num_layers.get())
		
		self.VECTOR_D = [10**(-7)*float(k) for k in self.entry_vector_d.get().split(',')]
		self.VECTOR_D[0] = -self.VECTOR_D[0]
		self.VECTOR_W = [float(k) for k in self.entry_vector_W.get().split(',')]
		for index in range(len(self.VECTOR_W)):
			self.VECTOR_W[index] = ((self.VECTOR_W[index]*u.eV).to(u.erg)).value

		self.VECTOR_MASS = [self.ME*float(k) for k in self.entry_vector_mass.get().split(',')]

		if not len(self.VECTOR_D)==self.NUM_LAYERS: 
			mb.showerror("Ошибка", "Количество слоёв не соответствует размерности вектора размеров слоёв!")
		if not len(self.VECTOR_W)==self.NUM_LAYERS:
			mb.showerror("Ошибка", "Размерность вектора потенциалов не равна количеству слоёв!")
		if not len(self.VECTOR_MASS)==self.NUM_LAYERS:
			mb.showerror("Ошибка", "Размерность вектора масс не равна количеству слоёв!")



	def find_roots(self, event):
		

		Task1 = EnergyFinding(self.NUM_LAYERS,self.VECTOR_D,self.VECTOR_W,self.VECTOR_MASS)
		self.ENERGY_ROOTS = sorted(Task1.find_roots())
		E_roots = []
		for ind in range(len(self.ENERGY_ROOTS)):
			E_roots.append((self.ENERGY_ROOTS[ind]*u.erg).to(u.eV).value)

		string_roots = 'Уровни энергии:\n'+' eV,\n'.join(map(str, E_roots)) + ' eV'
		self.txt.delete('1.0',END)
		self.txt.insert(END,string_roots)
		print()
		Task1.en_plot()


	def find_vawe_funct(self, event):

		Task1 = VaweFunctionFinding(self.NUM_LAYERS,self.VECTOR_D,self.VECTOR_W,self.VECTOR_MASS,self.ENERGY_ROOTS)
		Task1.vawe_func_plot(self.ENERGY_ROOTS)

	def find_norm_function(self, event):
		
		Task1 = VaweFunctionFinding(self.NUM_LAYERS,self.VECTOR_D,self.VECTOR_W,self.VECTOR_MASS,self.ENERGY_ROOTS)
		Task1.norm_funct_plot(int(self.entry_norm_index.get()))


class Exponential():
	def __init__(self,master):
		
		self.master = master
		self.master.geometry("800x600+200+200")
		self.master.title("EXP_Solver")

		self.NUM_LAYERS = 1
		self.VECTOR_D = []
		self.VECTOR_W = []
		self.VECTOR_MASS = []
		self.VECTOR_EXP = []
		self.TOLL = 10
		self.ENERGY_ROOTS = []
		self.ME = (c.m_e.to(u.g)).value

		self.label_num_layers = Label(self.master, width=20, font ='arial 18')
		self.label_num_layers['text'] = "Количество слоёв"
		self.label_vector_d = Label(self.master, width=20, font ='arial 18')
		self.label_vector_d['text'] = "Толщина слоёв, нм"
		self.label_vector_W = Label(self.master, width=20, font ='arial 18')
		self.label_vector_W['text'] = "Потенциал слоёв, эВ"
		self.label_vector_mass = Label(self.master,width=20, font ='arial 18')
		self.label_vector_mass['text'] = "Массы слоёв, m/m0"

		self.label_exp_layers = Label(self.master,width=20, font ='arial 18')
		self.label_exp_layers['text'] = "Номера exp слоёв:"
		self.label_toll = Label(self.master,width=20, font ='arial 18')
		self.label_toll['text'] = "Число точек разбиения:"
		
 
		self.entry_num_layers = Entry(self.master, width=20, font=15)
		self.entry_vector_d = Entry(self.master, width=20, font=15)
		self.entry_vector_W = Entry(self.master, width=20, font=15)
		self.entry_vector_mass = Entry(self.master, width=20, font=15)
		self.entry_exp_layers = Entry(self.master, width=15, font=15)
		self.entry_toll = Entry(self.master, width=5, font=15)

		self.but_Energy = Button(self.master,font=('Ubuntu',12),width=25,height=3, text='Рассчитать уровни энергии')
		self.but_VaweFunction = Button(self.master,font=('Ubuntu',12),width=25,height=3, text='Рассчитать волновые функции')
		self.but_Select = Button(self.master,font=('Ubuntu',12),width=25,height=3, text='Принять')
		self.but_One_Function_norm = Button(self.master,font=('Ubuntu',12),width=25,height=3, text='Нормированная в.ф.')
		
		self.txt = Text(self.master,height=5,width=30)

		self.label_num_layers.grid(row=0, column=0, pady = 5)
		self.label_vector_d.grid(row=1, column=0,pady = 5)
		self.label_vector_W.grid(row=2, column=0,pady = 5)
		self.label_vector_mass.grid(row=3, column=0,pady = 5)
		self.label_exp_layers.grid(row=1,column=3, pady=5)
		self.entry_exp_layers.grid(row=2,column=3, pady=5)
		self.entry_num_layers.grid(row=0, column=1, ipady = 3)
		self.entry_vector_d.grid(row=1, column=1, ipady = 3)
		self.entry_vector_W.grid(row=2, column=1, ipady = 3)
		self.entry_vector_mass.grid(row=3, column=1, ipady = 3)

		self.but_Select.grid(row=9, column=0)
		self.but_Energy.grid(row=10, column=0, pady = 35)
		self.but_VaweFunction.grid(row=11, column=0)
		self.label_toll.grid(row=12, column=0,pady=25)
		self.txt.grid(row=10,column=1)
		self.entry_toll.grid(row=12,column=1,pady=25)

		self.but_Energy.bind("<Button-1>",self.find_roots)
		self.but_VaweFunction.bind("<Button-1>", self.find_vawe_funct)
		self.but_Select.bind("<Button-1>",self.set_values)


	def set_values(self,event):
		
		self.NUM_LAYERS = int(self.entry_num_layers.get())
		self.TOLL = int(self.entry_toll.get())
		
		self.VECTOR_D = [10**(-7)*float(k) for k in self.entry_vector_d.get().split(',')]
		self.VECTOR_D[0] = -self.VECTOR_D[0]
		self.VECTOR_EXP = [int(n) for n in self.entry_exp_layers.get().split(',')]
		self.VECTOR_W = [float(k) for k in self.entry_vector_W.get().split(',')]
		for index in range(len(self.VECTOR_W)):
			self.VECTOR_W[index] = ((self.VECTOR_W[index]*u.eV).to(u.erg)).value

		self.VECTOR_MASS = [self.ME*float(k) for k in self.entry_vector_mass.get().split(',')]

		if not len(self.VECTOR_D)==self.NUM_LAYERS: 
			mb.showerror("Ошибка", "Количество слоёв не соответствует размерности вектора размеров слоёв!")
		if not len(self.VECTOR_W)==self.NUM_LAYERS:
			mb.showerror("Ошибка", "Размерность вектора потенциалов не равна количеству слоёв!")
		if not len(self.VECTOR_MASS)==self.NUM_LAYERS:
			mb.showerror("Ошибка", "Размерность вектора масс не равна количеству слоёв!")



	def find_roots(self, event):
		

		Task1 = EnergyFinding_exp(self.NUM_LAYERS,self.VECTOR_D,self.VECTOR_W,self.VECTOR_MASS,self.VECTOR_EXP,self.TOLL)
		self.ENERGY_ROOTS = sorted(Task1.find_roots_exp())
		E_roots = []
		for ind in range(len(self.ENERGY_ROOTS)):
			E_roots.append((self.ENERGY_ROOTS[ind]*u.erg).to(u.eV).value)

		string_roots = 'Уровни энергии:\n'+' eV,\n'.join(map(str, E_roots)) + ' eV'
		self.txt.delete('1.0',END)
		self.txt.insert(END,string_roots)
		print()
		Task1.en_plot_exp()


	def find_vawe_funct(self, event):

		Task1 = VaweFunctionFinding_exp(self.NUM_LAYERS,self.VECTOR_D,self.VECTOR_W,self.VECTOR_MASS,self.ENERGY_ROOTS,self.VECTOR_EXP,self.TOLL)
		Task1.vawe_func_plot_exp(self.ENERGY_ROOTS)

	def find_norm_function(self, event):
		
		Task1 = VaweFunctionFinding_exp(self.NUM_LAYERS,self.VECTOR_D,self.VECTOR_W,self.VECTOR_MASS,self.ENERGY_ROOTS,self.VECTOR_EXP,self.TOLL)
		Task1.norm_funct_plot_exp(int(self.entry_norm_index.get()))



def main():

	root = Tk()
	myGUIMain = Main(root)
	root.mainloop()

if __name__ == "__main__":
	main()