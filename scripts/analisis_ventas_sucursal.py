import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class AnalisisVentasSucursal:
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.df_alquileres = None
        self.df_sucursales = None
        self.df_vehiculos = None
        self.df_clientes = None
        self.df_empleados = None
        self.df_gastos = None
        
    def cargar_datos(self):
        """Carga todos los archivos CSV"""
        try:
            self.df_alquileres = pd.read_csv(f'{self.data_path}alquileres.csv')
            self.df_sucursales = pd.read_csv(f'{self.data_path}sucursales.csv')
            self.df_vehiculos = pd.read_csv(f'{self.data_path}vehiculos.csv')
            self.df_clientes = pd.read_csv(f'{self.data_path}clientes.csv')
            self.df_empleados = pd.read_csv(f'{self.data_path}empleados.csv')
            self.df_gastos = pd.read_csv(f'{self.data_path}gastos.csv')
            print("Datos cargados exitosamente")
        except Exception as e:
            print(f"Error al cargar datos: {e}")
    
    def preprocesar_datos(self):
        """Preprocesa y limpia los datos"""
        # Convertir fechas
        self.df_alquileres['fecha_inicio'] = pd.to_datetime(self.df_alquileres['fecha_inicio'])
        self.df_alquileres['fecha_fin'] = pd.to_datetime(self.df_alquileres['fecha_fin'])
        
        # Calcular duración del alquiler
        self.df_alquileres['duracion_dias'] = (self.df_alquileres['fecha_fin'] - self.df_alquileres['fecha_inicio']).dt.days
        
        # Extraer información temporal
        self.df_alquileres['año'] = self.df_alquileres['fecha_inicio'].dt.year
        self.df_alquileres['mes'] = self.df_alquileres['fecha_inicio'].dt.month
        self.df_alquileres['trimestre'] = self.df_alquileres['fecha_inicio'].dt.quarter
        
        print("Datos preprocesados exitosamente")
    
    def crear_datamart_ventas(self):
        """Crea el datamart específico para análisis de ventas por sucursal"""
        # Unir datos de alquileres con sucursales
        datamart = self.df_alquileres.merge(self.df_sucursales, on='id_sucursal', how='left')
        
        # Agregar información de vehículos
        datamart = datamart.merge(self.df_vehiculos[['id_vehiculo', 'tipo_vehiculo', 'marca', 'modelo']], 
                                 on='id_vehiculo', how='left')
        
        # Agregar información de clientes
        datamart = datamart.merge(self.df_clientes[['id_cliente', 'categoria_cliente', 'edad', 'genero']], 
                                 on='id_cliente', how='left')
        
        # Agregar información de empleados
        datamart = datamart.merge(self.df_empleados[['id_empleado', 'nombre', 'apellido', 'puesto']], 
                                 on='id_empleado', how='left')
        
        # Guardar datamart
        os.makedirs('data/processed', exist_ok=True)
        datamart.to_csv('data/processed/datamart_ventas_sucursal.csv', index=False)
        
        return datamart
    
    def analizar_ventas_por_sucursal(self, datamart):
        """Realiza análisis de ventas por sucursal"""
        # Análisis agregado por sucursal
        ventas_sucursal = datamart.groupby(['id_sucursal', 'nombre_sucursal', 'ciudad']).agg({
            'monto_total': ['sum', 'mean', 'count'],
            'monto_seguro': 'sum',
            'monto_combustible': 'sum',
            'duracion_dias': 'mean'
        }).round(2)
        
        # Aplanar columnas multinivel
        ventas_sucursal.columns = ['_'.join(col).strip() for col in ventas_sucursal.columns]
        ventas_sucursal = ventas_sucursal.reset_index()
        
        # Renombrar columnas
        ventas_sucursal.columns = ['id_sucursal', 'nombre_sucursal', 'ciudad', 
                                  'ingresos_totales', 'ingreso_promedio', 'num_alquileres',
                                  'total_seguros', 'total_combustible', 'duracion_promedio']
        
        # Calcular métricas adicionales
        ventas_sucursal['ingreso_por_dia'] = ventas_sucursal['ingresos_totales'] / ventas_sucursal['duracion_promedio']
        
        # Guardar resultados
        ventas_sucursal.to_csv('data/processed/analisis_ventas_sucursal.csv', index=False)
        
        return ventas_sucursal
    
    def generar_visualizaciones(self, ventas_sucursal):
        """Genera visualizaciones del análisis"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico 1: Ingresos totales por sucursal
        axes[0,0].bar(ventas_sucursal['nombre_sucursal'], ventas_sucursal['ingresos_totales'])
        axes[0,0].set_title('Ingresos Totales por Sucursal')
        axes[0,0].set_ylabel('Ingresos (€)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Número de alquileres por sucursal
        axes[0,1].bar(ventas_sucursal['nombre_sucursal'], ventas_sucursal['num_alquileres'], color='orange')
        axes[0,1].set_title('Número de Alquileres por Sucursal')
        axes[0,1].set_ylabel('Cantidad')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Gráfico 3: Ingreso promedio por alquiler
        axes[1,0].bar(ventas_sucursal['nombre_sucursal'], ventas_sucursal['ingreso_promedio'], color='green')
        axes[1,0].set_title('Ingreso Promedio por Alquiler')
        axes[1,0].set_ylabel('Ingresos (€)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Gráfico 4: Duración promedio de alquileres
        axes[1,1].bar(ventas_sucursal['nombre_sucursal'], ventas_sucursal['duracion_promedio'], color='red')
        axes[1,1].set_title('Duración Promedio de Alquileres')
        axes[1,1].set_ylabel('Días')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/processed/analisis_ventas_visualizacion.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generar_reporte(self, ventas_sucursal):
        """Genera reporte de análisis"""
        reporte = f"""
# REPORTE DE ANÁLISIS DE VENTAS POR SUCURSAL - RENT4YOU

## Resumen Ejecutivo
Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Métricas Principales

### Ranking de Sucursales por Ingresos:
{ventas_sucursal.sort_values('ingresos_totales', ascending=False)[['nombre_sucursal', 'ciudad', 'ingresos_totales']].to_string(index=False)}

### Estadísticas Generales:
- Total de sucursales analizadas: {len(ventas_sucursal)}
- Ingresos totales de la red: €{ventas_sucursal['ingresos_totales'].sum():,.2f}
- Promedio de ingresos por sucursal: €{ventas_sucursal['ingresos_totales'].mean():,.2f}
- Total de alquileres: {ventas_sucursal['num_alquileres'].sum()}

### Sucursal con Mejor Performance:
- Mayores ingresos: {ventas_sucursal.loc[ventas_sucursal['ingresos_totales'].idxmax(), 'nombre_sucursal']}
- Mayor número de alquileres: {ventas_sucursal.loc[ventas_sucursal['num_alquileres'].idxmax(), 'nombre_sucursal']}
- Mayor ingreso promedio: {ventas_sucursal.loc[ventas_sucursal['ingreso_promedio'].idxmax(), 'nombre_sucursal']}

## Recomendaciones:
1. Analizar las mejores prácticas de la sucursal con mayores ingresos
2. Implementar estrategias de mejora en sucursales con menor performance
3. Optimizar la duración de alquileres para maximizar ingresos
        """
        
        with open('data/processed/reporte_analisis.md', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        return reporte

def main():
    # Crear instancia del analizador
    analizador = AnalisisVentasSucursal()
    
    # Ejecutar análisis completo
    analizador.cargar_datos()
    analizador.preprocesar_datos()
    
    # Crear datamart
    datamart = analizador.crear_datamart_ventas()
    print(f"Datamart creado con {len(datamart)} registros")
    
    # Realizar análisis
    ventas_sucursal = analizador.analizar_ventas_por_sucursal(datamart)
    print("\nAnálisis de ventas por sucursal:")
    print(ventas_sucursal)
    
    # Generar visualizaciones
    analizador.generar_visualizaciones(ventas_sucursal)
    
    # Generar reporte
    reporte = analizador.generar_reporte(ventas_sucursal)
    print("\nReporte generado exitosamente")
    
    print("\nArchivos generados en data/processed/:")
    print("- datamart_ventas_sucursal.csv")
    print("- analisis_ventas_sucursal.csv")
    print("- analisis_ventas_visualizacion.png")
    print("- reporte_analisis.md")

if __name__ == "__main__":
    main()