from types import MappingProxyType  # неизменяемый словарь
import numpy as np
from numpy import nan, isnan, pi, sqrt, array, linspace, arange, radians
from scipy import interpolate, integrate
import matplotlib.pyplot as plt

from material import Material
from airfoil import Airfoil

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

    @classmethod
    def help(cls):
        version = '3.0'
        print('Расчет на прочность')
        return version

    def __init__(self, material: Material, sections: dict[float | int | np.number: list, tuple, np.ndarray]) -> None:
        # проверка на тип данных material
        assert isinstance(material, Material)

        assert isinstance(sections, dict)
        assert all(isinstance(key, (int, float, np.number)) for key in sections.keys())
        assert len(sections) >= 1  # min количество сечений
        assert all(isinstance(value, (list, tuple, np.ndarray)) for value in sections.values())
        assert all(isinstance(coord, (list, tuple, np.ndarray)) for value in sections.values() for coord in value)
        assert all(len(coord) == 2 for value in sections.values() for coord in value)  # x, y
        assert all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                   for el in sections.values() for x, y in el)

        self.__material = material
        self.__sections = dict(sorted(sections.items(), key=lambda item: item[0]))  # сортировка по высоте

        self.__height = max(self.__sections.keys()) - min(self.__sections.keys())

    @property
    def material(self):
        return self.__material

    @property
    def height(self) -> float:
        return self.__height

    def show(self, D: int, **kwargs):
        """Визуализация"""
        assert isinstance(D, int) and D in (2, 3)  # мерность пространства

        if 2 == D:
            plt.figure(figsize=kwargs.pop('figsize', (8, 8)))
            plt.axis('equal')
            plt.grid(True)
            for i, (r, section) in enumerate(self.__sections.items()):
                plt.plot(*array(section, dtype='float16').T,
                         color='black', ls='solid', linewidth=(1 + 2 / (len(self.__sections) - 1) * i))

        elif 3 == D:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            plt.figure(figsize=kwargs.pop('figsize', (8, 8)))
            ax = plt.axes(projection='3d')
            ax.axis('equal')
            for z, section in self.__sections.items():
                x, y = array(section, dtype='float16').T
                vertices = [list(zip(x, y, [z] * len(x)))]
                poly = Poly3DCollection(vertices, color='black', alpha=0.8)
                ax.add_collection3d(poly)
            ax.set_title(kwargs.pop('title', 'Blade'), fontsize=14, fontweight='bold')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_zlabel('z', fontsize=12)

        plt.show()

    def tensions(self, show=True):
        """Расчет на прочность"""

        if show: self.__show_tensions()
        return

    def __show_tensions(self):
        """Визуализация расчет на прочность"""

        plt.figure(figsize=(12, 8))
        plt.show()

    def natural_frequencies(self, max_k: int) -> tuple[float, str]:
        """Частота собственных колебаний [6]"""
        self.I = self.b * self.c ** 3 / 12  # момент инерции сечения
        self.F = self.b * self.c  # площадь сечения

        if "крепление в заделке":
            k = array([1.875, 4.694] + [pi * (i - 0.5) for i in range(3, max_k + 1, 1)])
        elif "крепление шарнирное":
            k = array([0] + [pi * (i - 0.75) for i in range(2, max_k + 1, 1)])
        else:
            raise

        f = self.material.E(0) * self.I / (self.material.density(0) * np.mean(self.F))
        f = sqrt(f)
        f *= k ** 2 / (2 * pi * self.h ** 2)
        return float(f), '1/s'

    def campbell_diagram(self, max_rotation_frequency: int, k=arange(1, 11, 1), **kwargs):
        """Диаграмма Кэмпбелла [6]"""

        rotation_frequency = arange(0, max_rotation_frequency + 1, 1) * (2 * pi)  # перевод из рад/c в 1/c=об/c=Гц
        # динамическая частота колебаний вращающегося диска
        # self.R[0] = радиус корня
        f = sqrt(self.natural_frequencies(max(k))[0] ** 2 + (1 + self.R[0] / self.h) * rotation_frequency ** 2)
        resonance = set()  # резонансные частоты [1/с]

        plt.figure(figsize=kwargs.pop('figsize', (8, 8)))
        plt.title('Campbell diagram', fontsize=14, fontweight='bold')
        for k_ in k:
            plt.plot([0, rotation_frequency[-1]], [0, rotation_frequency[-1] * k_],
                     color='orange', linestyle='solid', linewidth=1)
            plt.text(rotation_frequency[-1], rotation_frequency[-1] * k_, f'k{k_}',
                     fontsize=12, ha='left', va='center')
            if k_ ** 2 - (1 + self.R[0] / self.h) >= 0:
                x0 = f[0] / sqrt(k_ ** 2 - (1 + self.R[0] / self.h))  # f
                if not isnan(x0) and x0 <= rotation_frequency[-1]:
                    resonance.add(round(x0, 6))
                    plt.scatter(x0, k_ * x0, color='red')
        plt.plot(rotation_frequency, f, color='black', linestyle='solid', linewidth=2, label='f')
        plt.xlabel('Frequency [1/s]', fontsize=12)
        plt.ylabel('Frequency [1/s]', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.show()

        return sorted(list(map(float, resonance)), reverse=False), '1/s'


def test():
    """Тестирование библиотеки"""
    print(Blade.help())

    blades = list()

    if 1:
        material = Material('ЖС-40', {'density': 8600})

        airfoil0 = Airfoil('BMSTU', 40, 1, radians(20),
                           rotation_angle=radians(60), relative_inlet_radius=0.06, relative_outlet_radius=0.03,
                           inlet_angle=radians(20), outlet_angle=radians(15), x_ray_cross=0.35, upper_proximity=0.5)
        airfoil1 = Airfoil('BMSTU', 40, 1, radians(30),
                           rotation_angle=radians(50), relative_inlet_radius=0.04, relative_outlet_radius=0.025,
                           inlet_angle=radians(15), outlet_angle=radians(10), x_ray_cross=0.35, upper_proximity=0.5)
        airfoil2 = Airfoil('BMSTU', 40, 1, radians(40),
                           rotation_angle=radians(40), relative_inlet_radius=0.03, relative_outlet_radius=0.02,
                           inlet_angle=radians(10), outlet_angle=radians(5), x_ray_cross=0.35, upper_proximity=1)

        sections = {0.5: airfoil0.transform(airfoil0.coordinates,
                                            x0=airfoil0.properties['x0'], y0=airfoil0.properties['y0']),
                    0.6: airfoil1.transform(airfoil1.coordinates,
                                            x0=airfoil1.properties['x0'], y0=airfoil1.properties['y0']),
                    0.7: airfoil2.transform(airfoil2.coordinates,
                                            x0=airfoil2.properties['x0'], y0=airfoil2.properties['y0'])}

        blade = Blade(material=material, sections=sections)
        blades.append(blade)

    for blade in blades:
        blade.show(2)
        blade.show(3)

        tensions = blade.tensions(show=True)


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
