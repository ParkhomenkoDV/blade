from types import MappingProxyType  # неизменяемый словарь
import numpy as np
from numpy import nan, isnan, pi, sqrt, exp, log as ln, array, linspace, arange, radians, cos, sin, atan
from scipy import interpolate, integrate
import matplotlib.pyplot as plt

from decorators import timeit
from material import Material
from foil import Foil

# Список использованной литературы
REFERENCES = MappingProxyType({
    1: '''Елисеев Ю.С., Крымов В.В., Манушин Э.А. и др.
Конструирование и расчет на прочность турбомашин ГТ и КУ:
Учебник для студентов вузов / Под общей ред. М.И. Осипова. – М.: МГТУ, 2009''',
    2: '''Иноземцев А.А. Динамика и прочность авиационных двигателей и энергетических установок:
учебник / А.А. Иноземцев, М.А. Нихамкин, В.Л. Сандрацкий. – М.: 
Машиностроение, 2008. – Т.2. – 368 с.; Т.4. – 192 с.''',
    3: '''Костюк А.Г. Динамика и прочность турбомашин:
Учебник для вузов. – 2-е изд., перераб. и доп. – М.: Издательство МЭИ, 2000. – 480 с.''',
    4: '''Щегляев А.В. Паровые турбины. – М.: Энергоатомиздат, 1993. – В двух книгах.''',
    5: '''Малинин Н.Н. Прочность турбомашин. – М.: Машиностроние, 1962. – 291 с.''',
    6: '''Колебания / В.С. Чигрин. - Пособие по лабораторному практикуму - Рыбинск: РГАТА, 2005. -20 с.''',
})


class Blade:
    """Лопатка/винт/лопасть"""

    __slots__ = ('__material', '__sections', '__bondages',  # необходимые параметры
                 '__start_point', '__center_pressure', '__height', '__volume',  # производные параметры
                 '__f_area', '__f_center_pressure', '__f_installation_angle')  # интерполированные параметры

    @timeit(4)
    def __init__(self, material: Material, sections: dict[float | int | np.number: list, tuple, np.ndarray],
                 bondages: tuple[dict] | list[dict] = tuple()) -> None:
        # проверка на тип данных material
        assert isinstance(material, Material)
        self.__material = material

        assert isinstance(sections, dict), 'type(sections) is not dict'
        assert all(isinstance(key, (int, float, np.number)) for key in sections.keys())
        assert len(sections) >= 2  # min количество сечений для определения высоты
        assert all(isinstance(value, (list, tuple, np.ndarray)) for value in sections.values())
        assert all(isinstance(coord, (list, tuple, np.ndarray)) for value in sections.values() for coord in value)
        assert all(len(coord) == 2 for value in sections.values() for coord in value)  # x, y
        assert all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                   for el in sections.values() for x, y in el)
        sections = dict(sorted(sections.items(), key=lambda item: item[0]))  # сортировка сечений по высоте
        self.__sections, self.__start_point, self.__center_pressure = dict(), dict(), dict()
        for radius, section in sections.items():  # TODO: multiprocessing
            self.__start_point[radius] = section[0]  # точка старта отсчета профиля против часовой стрелки = вых кромка
            foil = Foil.load(section, deg=1, discreteness=len(section), name=f'foil_{radius}'.replace('.', '_'))
            center_pressure = foil.center_pressure(0.3)  # расчет профиля
            center_pressure = Foil.transform([center_pressure], scale=foil.chord)
            center_pressure = Foil.transform(center_pressure, angle=foil.installation_angle)
            center_pressure = Foil.transform(center_pressure, x0=-section[0][0], y0=-section[0][1])
            self.__center_pressure[radius] = center_pressure[0]
            self.__sections[radius] = foil

        assert isinstance(bondages, (tuple, list))
        assert all(isinstance(bondage, dict) for bondage in bondages)
        assert all('radius' in bondage.keys() and 'volume' in bondage.keys() for bondage in bondages)
        assert all(isinstance(bondage['radius'], (int, float, np.number)) and
                   isinstance(bondage['volume'], (int, float, np.number)) for bondage in bondages)
        assert all(0 <= bondage['radius'] and 0 <= bondage['volume'] for bondage in bondages)
        self.__bondages = bondages  # бондажи

        self.__f_area = interpolate.interp1d(list(self.__sections.keys()),
                                             [foil.properties['area'] * foil.chord ** 2
                                              for foil in self.__sections.values()],
                                             kind=1, fill_value='extrapolate')
        self.__f_center_pressure = (interpolate.interp1d(list(self.__sections.keys()),
                                                         [self.__center_pressure[radius][0]
                                                          for radius in self.__sections],
                                                         kind=1, fill_value='extrapolate'),
                                    interpolate.interp1d(list(self.__sections.keys()),
                                                         [self.__center_pressure[radius][1]
                                                          for radius in self.__sections],
                                                         kind=1, fill_value='extrapolate'))
        self.__f_installation_angle = interpolate.interp1d(list(self.__sections.keys()),
                                                           [foil.installation_angle
                                                            for foil in self.__sections.values()],
                                                           kind=1, fill_value='extrapolate')

    @property
    def material(self) -> Material:
        return self.__material

    @property
    def sections(self) -> dict:
        return self.__sections

    @property
    def bondages(self) -> dict:
        return self.__bondages

    @property
    def height(self) -> float:
        if hasattr(self, '_Blade__height'): return self.__height
        radius0, *_, radius1 = tuple(self.__sections.keys())
        self.__height = radius1 - radius0
        return self.__height

    @property
    def volume(self) -> float:
        if hasattr(self, '_Blade__volume'): return self.__volume
        radius0, *_, radius1 = tuple(self.__sections.keys())
        self.__volume = integrate.quad(self.__f_area, radius0, radius1)[0]
        return self.__volume

    def mass(self, temperature: float | int | np.number) -> float:
        assert isinstance(temperature, (float, int, np.number))
        assert 0 < temperature
        return float(self.material.density(temperature) * self.volume)

    def show(self, D: int, **kwargs) -> None:
        """Визуализация"""
        assert D in (2, 3)  # мерность пространства

        plt.figure(figsize=kwargs.pop('figsize', (8, 8)))

        if 2 == D:
            plt.title(kwargs.pop('title', 'Blade'), fontsize=14, fontweight='bold')
            plt.axis('equal')
            plt.grid(True)
            for i, (r, foil) in enumerate(self.sections.items()):
                coordinates = Foil.transform(foil.relative_coordinates, scale=foil.chord)  # масштабирование
                coordinates = Foil.transform(coordinates, angle=foil.installation_angle)  # поворот
                x, y = array(coordinates, dtype='float32').T
                plt.plot(x - self.__start_point[r][0], y - self.__start_point[r][1],
                         color=kwargs.get('color', 'black'), ls='solid',
                         linewidth=(1 + 2 / (len(self.sections) - 1) * i))
            plt.xlabel('x', fontsize=12), plt.ylabel('y', fontsize=12)

        elif 3 == D:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            ax = plt.axes(projection='3d')
            ax.set_title(kwargs.pop('title', 'Blade'), fontsize=14, fontweight='bold')
            ax.axis('equal')
            for r, foil in self.sections.items():
                coordinates = Foil.transform(foil.relative_coordinates, scale=foil.chord)  # масштабирование
                coordinates = Foil.transform(coordinates, angle=foil.installation_angle)  # поворот
                x, y = array(coordinates, dtype='float32').T
                vertices = [list(zip(x - self.__start_point[r][0], y - self.__start_point[r][1], [r] * len(x)))]
                poly = Poly3DCollection(vertices, color=kwargs.get('color', 'black'), alpha=0.8)
                ax.add_collection3d(poly)
            ax.set_xlabel('x', fontsize=12), ax.set_ylabel('y', fontsize=12), ax.set_zlabel('z', fontsize=12)

        plt.tight_layout()
        plt.show()

    @property
    def radius_equal_strength(self) -> float:
        """Радиус равнопрочности"""
        radius0, *_, radius1 = tuple(self.__sections.keys())  # радиус втулки и периферии
        area0, *_, area1 = self.__f_area(radius0), self.__f_area(radius1)  # площадь сечений втулки и периферии
        return float(sqrt((radius0 ** 2 - radius1 ** 2 * ln(area1 / area0)) / (1 - ln(area1 / area0))))

    def radial_force(self, rotation_frequency: float | int | np.number,
                     temperature: float | int | np.number):
        assert isinstance(rotation_frequency, (int, float, np.number))
        assert isinstance(temperature, (int, float, np.number)) and 0 < temperature

        radius1 = tuple(self.sections.keys())[-1]  # радиус периферии

        return lambda z: float(self.material.density(temperature) * rotation_frequency ** 2 *
                               (integrate.quad(lambda r: self.__f_area(r) * r, z, radius1)[0] +
                                sum([b['radius'] * b['volume'] for b in self.bondages])))

    def radial_tension(self, rotation_frequency: float | int | np.number,
                       temperature: float | int | np.number):
        assert isinstance(rotation_frequency, (int, float, np.number))
        assert isinstance(temperature, (int, float, np.number)) and 0 < temperature

        return lambda z: float(self.radial_force(rotation_frequency, temperature)(z) / self.__f_area(z))

    def area_equal_strength(self, rotation_frequency: float | int | np.number):
        assert isinstance(rotation_frequency, (int, float, np.number))

        radius1 = tuple(self.sections.keys())[-1]  # радиус периферии
        area1 = self.__f_area(radius1)  # площадь периферии

        return lambda z: float(area1 * exp((self.radius_equal_strength ** 2 - z ** 2) /
                                           (radius1 ** 2 - self.radius_equal_strength ** 2))) \
            if z <= self.radius_equal_strength else area1

    def radial_force_equal_strength(self, rotation_frequency: float | int | np.number,
                                    temperature: float | int | np.number):
        assert isinstance(rotation_frequency, (int, float, np.number))
        assert isinstance(temperature, (int, float, np.number)) and 0 < temperature

        radius1 = tuple(self.__sections.keys())[-1]  # радиус периферии

        return lambda z: \
            float(self.material.density(temperature) * rotation_frequency ** 2 *
                  (integrate.quad(lambda r: self.area_equal_strength(rotation_frequency)(r) * r, z, radius1)[0] +
                   sum([b['radius'] * b['volume'] for b in self.bondages])))

    def radial_tension_equal_strength(self, rotation_frequency: float | int | np.number,
                                      temperature: float | int | np.number):
        assert isinstance(rotation_frequency, (int, float, np.number))
        assert isinstance(temperature, (int, float, np.number)) and 0 < temperature

        return lambda z: (self.radial_force_equal_strength(rotation_frequency, temperature)(z) /
                          self.area_equal_strength(rotation_frequency)(z))

    def show_equal_strength(self, rotation_frequency: float | int | np.number,
                            temperature: float | int | np.number, discreteness: int = 100, **kwargs) -> None:
        """Визуализация равнопрочности"""
        assert isinstance(rotation_frequency, (int, float, np.number))
        assert isinstance(temperature, (int, float, np.number)) and 0 < temperature
        assert isinstance(discreteness, (int, np.integer)) and 10 <= discreteness

        radius0, *_, radius1 = tuple(self.sections.keys())  # радиус втулки и периферии
        radius = linspace(radius0, radius1, discreteness, endpoint=True)

        fg = plt.figure(figsize=kwargs.pop('figsize', (16, 8)))
        gs = fg.add_gridspec(nrows=1, ncols=3)
        plt.suptitle('Equal strength', fontsize=16, fontweight='bold')

        fg.add_subplot(gs[0, 0])
        plt.plot(self.__f_area(radius), radius, color='black', ls='solid', linewidth=2, label='fact')
        plt.plot([self.area_equal_strength(rotation_frequency)(r) for r in radius], radius,
                 color='green', ls='solid', linewidth=2, label='equal strength')
        plt.grid(True)
        plt.xlabel('area', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        fg.add_subplot(gs[0, 1])
        plt.plot([self.radial_force(rotation_frequency, temperature)(r) for r in radius], radius,
                 color='black', ls='solid', linewidth=2, label='fact')
        plt.plot([self.radial_force_equal_strength(rotation_frequency, temperature)(r) for r in radius], radius,
                 color='blue', ls='solid', linewidth=2, label='equal strength')
        plt.grid(True)
        plt.xlabel('radial force', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        fg.add_subplot(gs[0, 2])
        plt.plot([self.radial_tension(rotation_frequency, temperature)(r) for r in radius], radius,
                 color='black', ls='solid', linewidth=2, label='fact')
        plt.plot([self.radial_tension_equal_strength(rotation_frequency, temperature)(r) for r in radius], radius,
                 color='red', ls='solid', linewidth=2, label='equal strength')
        plt.grid(True)
        plt.xlabel('radial tension', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def __validate_parameter(parameter: dict, check_positive: bool = True) -> None:
        """Проверка верности введенных значений термодинамического параметра"""
        assert isinstance(parameter, dict)
        assert 0 < len(parameter)
        for radius, value in parameter.items():
            assert isinstance(radius, (float, int, np.number))
            assert isinstance(value, (float, int, np.number))
            assert 0 <= radius
            if check_positive: assert 0 < value

    @staticmethod
    def __interpolate_parameter(parameter: dict, deg: int = 1) -> interpolate.interp1d:
        """Интерполирование параметра"""
        parameter = dict(sorted(parameter.items(), key=lambda item: item[0], reverse=False))
        return interpolate.interp1d(tuple(parameter.keys()), tuple(parameter.values()),
                                    kind=min(len(parameter) - 1, deg), fill_value='extrapolate')

    # TODO: Расчет на прочность
    def tensions(self, amount: int | np.integer, rotation_frequency: float | int | np.number,
                 density: tuple[dict] = tuple(),
                 pressure: tuple[dict] = tuple(),
                 velocity_axial: tuple[dict] = tuple(), velocity_tangential: tuple[dict] = tuple(),
                 deg: int = 1,
                 show: bool = True):
        """Расчет на прочность"""
        assert isinstance(amount, (int, np.integer)) and 1 <= amount
        assert isinstance(rotation_frequency, (float, int, np.number))
        assert isinstance(deg, (int, np.integer)) and 1 <= deg <= 3

        functions = dict()

        for name, parameters, check_positive in (('density', density, True),
                                                 ('pressure', pressure, True),
                                                 ('velocity_axial', velocity_axial, False),
                                                 ('velocity_tangential', velocity_tangential, False)):
            assert isinstance(parameters, (tuple, list)), f'{name} is not list or tuple'
            assert len(parameters) >= 2, f'len({name}) < 2'

            functions[name] = list()
            for parameter in [parameters[0], parameters[-1]]:  # для ускорения и ненадобности
                self.__validate_parameter(parameter, check_positive=check_positive)
                functions[name].append(self.__interpolate_parameter(parameter, deg=deg))

        radius0, *_, radius1 = tuple(self.sections.keys())

        f_density_i, f_density_o = functions['density']
        f_pressure_i, f_pressure_o = functions['pressure']
        f_velocity_axial_i, f_velocity_axial_o = functions['velocity_axial']
        f_velocity_tangential_i, f_velocity_tangential_o = functions['velocity_tangential']

        f_density = lambda z: (f_density_i(z) + f_density_o(z)) / 2  # средняя арифметическая плотность

        qx = lambda z: (2 * pi * z / amount *
                        ((f_pressure_i(z) - f_pressure_o(z)) -
                         f_density(z) *
                         f_velocity_axial_i(z) * (f_velocity_axial_o(z) - f_velocity_axial_i(z))))
        qy = lambda z: (2 * pi * z / amount *
                        f_density(z) *
                        f_velocity_axial_i(z) * (f_velocity_tangential_o(z) - f_velocity_tangential_i(z)))

        f_moment = lambda q, z: integrate.quad(lambda zz: q(zz) * (zz - z), z, radius1)[0]

        moment_x = lambda z: integrate.quad(lambda zz: qx(zz) * (zz - z), z, radius1)[0]
        moment_y = lambda z: -integrate.quad(lambda zz: qy(zz) * (zz - z), z, radius1)[0]

        moment_torque = lambda z: (
            integrate.quad(lambda zz:
                           qy(zz) * self.__f_center_pressure[0](zz) - qx(zz) * self.__f_center_pressure[1](zz),
                           z, radius1))[0]

        moment_xx = lambda z: 0 + moment_x(z)
        moment_yy = lambda z: 0 + moment_x(z)

        moment_u = lambda z: (moment_xx(z) * cos(self.__f_installation_angle(z)) +
                              moment_yy(z) * sin(self.__f_installation_angle(z)))
        moment_v = lambda z: (-moment_xx(z) * sin(self.__f_installation_angle(z)) +
                              moment_yy(z) * cos(self.__f_installation_angle(z)))

        for radius, foil in self.__sections.items():
            tau = moment_torque(radius) / (foil.properties['Wp'] * foil.chord ** 3)
            print(f'{tau = }')
            mu = moment_u(radius)
            mv = moment_v(radius)
            betta = atan((mv / mu) * (foil.properties['Jx'] / foil.properties['Jy']))
            print(f'{betta = }')
            x_vynos = 0
            y_vynos = 0

        if show: self.__show_tensions()
        return

    def __show_tensions(self, discreteness: int = 100, **kwargs) -> None:
        """Визуализация расчет на прочность"""

        radius0, *_, radius1 = tuple(self.sections.keys())  # радиус втулки и периферии
        radius = linspace(radius0, radius1, discreteness, endpoint=True)

        fg = plt.figure(figsize=kwargs.pop('figsize', (18, 8)))
        gs = fg.add_gridspec(nrows=1, ncols=6)
        plt.suptitle('Tensions', fontsize=16, fontweight='bold')

        fg.add_subplot(gs[0, 0])
        #
        plt.grid(True)
        plt.xlabel('density', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        fg.add_subplot(gs[0, 1])
        #
        plt.grid(True)
        plt.xlabel('pressure', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        fg.add_subplot(gs[0, 2])
        #
        plt.grid(True)
        plt.xlabel('velocity', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        fg.add_subplot(gs[0, 3])
        plt.plot(self.__f_area(radius), radius, color='black', ls='solid', linewidth=2)
        plt.grid(True)
        plt.xlabel('area', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        fg.add_subplot(gs[0, 4])
        #
        plt.grid(True)
        plt.xlabel('force', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        fg.add_subplot(gs[0, 5])
        #
        plt.grid(True)
        plt.xlabel('tension', fontsize=12), plt.ylabel('radius', fontsize=12)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def natural_frequencies(self, radius: int, max_k: int) -> tuple[float, str]:
        """Частота собственных колебаний [6]"""
        self.I = self.b * self.c ** 3 / 12  # момент инерции сечения
        self.F = self.b * self.c  # площадь сечения

        if radius == 0:  # крепление в заделке
            k = array([1.875, 4.694] + [pi * (i - 0.5) for i in range(3, max_k + 1, 1)])
        elif radius == -1:  # крепление шарнирное
            k = array([0] + [pi * (i - 0.75) for i in range(2, max_k + 1, 1)])
        else:
            raise Exception(f'radius = 0 (крепление в заделке) or radius = -1 (крепление шарнирное)')

        f = self.material.E(0) * self.I / (self.material.density(0) * np.mean(self.F))
        f = sqrt(f)
        f *= k ** 2 / (2 * pi * self.h ** 2)
        return float(f), '1/s'

    def campbell_diagram(self, max_rotation_frequency: int, multiplicity=arange(1, 11, 1), **kwargs):
        """Диаграмма Кэмпбелла [6]"""

        rotation_frequency = arange(0, max_rotation_frequency + 1, 1) * (2 * pi)  # перевод из рад/c в 1/c=об/c=Гц
        # динамическая частота колебаний вращающегося диска
        # self.R[0] = радиус корня
        f = sqrt(
            self.natural_frequencies(max(multiplicity))[0] ** 2 + (1 + self.R[0] / self.h) * rotation_frequency ** 2)
        resonance = set()  # резонансные частоты [1/с]

        plt.figure(figsize=kwargs.pop('figsize', (8, 8)))
        plt.title('Campbell diagram', fontsize=14, fontweight='bold')
        for k in multiplicity:
            plt.plot([0, rotation_frequency[-1]], [0, rotation_frequency[-1] * k],
                     color='orange', linestyle='solid', linewidth=1)
            plt.text(rotation_frequency[-1], rotation_frequency[-1] * k, f'k{k}',
                     fontsize=12, ha='left', va='center')
            if k ** 2 - (1 + self.R[0] / self.h) >= 0:
                x0 = f[0] / sqrt(k ** 2 - (1 + self.R[0] / self.h))  # f
                if not isnan(x0) and x0 <= rotation_frequency[-1]:
                    resonance.add(round(x0, 6))
                    plt.scatter(x0, k * x0, color='red')
        plt.plot(rotation_frequency, f, color='black', linestyle='solid', linewidth=2, label='f')
        plt.xlabel('Frequency [1/s]', fontsize=12), plt.ylabel('Frequency [1/s]', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        return sorted(list(map(float, resonance)), reverse=False), '1/s'


def test():
    """Тестирование библиотеки"""

    blades = list()

    if 1:
        material = Material('ЖС-40', {'density': 8_600})

        foil0 = Foil('BMSTU', 40, 1, radians(20),
                     rotation_angle=radians(60), relative_inlet_radius=0.06, relative_outlet_radius=0.03,
                     inlet_angle=radians(20), outlet_angle=radians(15), x_ray_cross=0.35, upper_proximity=0.5)
        foil1 = Foil('BMSTU', 40, 1, radians(30),
                     rotation_angle=radians(50), relative_inlet_radius=0.04, relative_outlet_radius=0.025,
                     inlet_angle=radians(15), outlet_angle=radians(10), x_ray_cross=0.35, upper_proximity=0.5)
        foil2 = Foil('BMSTU', 40, 1, radians(40),
                     rotation_angle=radians(40), relative_inlet_radius=0.03, relative_outlet_radius=0.02,
                     inlet_angle=radians(10), outlet_angle=radians(5), x_ray_cross=0.35, upper_proximity=1)

        scale = 0.05  # [м]
        sections = {0.50: foil0.transform(foil0.coordinates,
                                          x0=foil0.properties['x0'], y0=foil0.properties['y0'], scale=scale),
                    0.55: foil1.transform(foil1.coordinates,
                                          x0=foil1.properties['x0'], y0=foil1.properties['y0'], scale=scale * 0.9),
                    0.60: foil2.transform(foil2.coordinates,
                                          x0=foil2.properties['x0'], y0=foil2.properties['y0'], scale=scale * 0.8)}

        blade = Blade(material=material, sections=sections)
        blades.append(blade)

    for blade in blades:
        blade.show(2)
        blade.show(3)

        rotation_frequency = 800
        temperature = 1_200

        print(f'{blade.height = }')
        print(f'{blade.volume = }')
        print(f'{blade.mass(temperature) = }')
        print()
        print(f'{blade.radius_equal_strength = }')
        print(f'{blade.radial_force(rotation_frequency, temperature)(0.5) = }')
        print(f'{blade.radial_tension(rotation_frequency, temperature)(0.5) = }')
        print(f'{blade.area_equal_strength(rotation_frequency)(0.5) = }')
        print(f'{blade.radial_force_equal_strength(rotation_frequency, temperature)(0.5) = }')
        print(f'{blade.radial_tension_equal_strength(rotation_frequency, temperature)(0.5) = }')

        blade.show_equal_strength(rotation_frequency, temperature)

        pressure_inlet = {0.50: 10 ** 5,
                          0.55: 10 ** 5,
                          0.60: 10 ** 5, }
        pressure_outlet = {0.50: 10 ** 5,
                           0.55: 10 ** 5,
                           0.60: 10 ** 5, }

        density_inlet = {0.50: 10 ** 5,
                         0.55: 10 ** 5,
                         0.60: 10 ** 5, }
        density_outlet = {0.50: 10 ** 5,
                          0.55: 10 ** 5,
                          0.60: 10 ** 5, }

        velocity_axial_inlet = {0.50: 10 ** 5,
                                0.55: 10 ** 5,
                                0.60: 10 ** 5, }
        velocity_axial_outlet = {0.50: 10 ** 5,
                                 0.55: 10 ** 5,
                                 0.60: 10 ** 5, }

        velocity_tangential_inlet = {0.50: 10 ** 5,
                                     0.55: 10 ** 5,
                                     0.60: 10 ** 5, }
        velocity_tangential_outlet = {0.50: 10 ** 5,
                                      0.55: 10 ** 5,
                                      0.60: 10 ** 5, }

        tensions = blade.tensions(44, 2_800,
                                  density=(density_inlet, density_outlet),
                                  pressure=(pressure_inlet, pressure_outlet),
                                  velocity_axial=(velocity_axial_inlet, velocity_axial_outlet),
                                  velocity_tangential=(velocity_tangential_inlet, velocity_tangential_outlet),
                                  deg=1,
                                  show=True)


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
