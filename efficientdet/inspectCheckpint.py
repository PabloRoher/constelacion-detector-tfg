# inspect_checkpoint.py
import tensorflow.compat.v1 as tf
import os

# --- CONFIGURACIÓN ---
# Apunta al archivo .meta de tu checkpoint. Cualquiera de los que tienes vale.
# Vamos a usar el último y mejor.
CHECKPOINT_PATH = 'automl/model_dir/model.ckpt-10256'

# --- SCRIPT ---
def inspect_variables_in_checkpoint(ckpt_path):
    print(f"Inspeccionando variables en el checkpoint: {ckpt_path}")
    try:
        # tf.train.list_variables enumera todas las variables (nombre, forma)
        variables = tf.train.list_variables(ckpt_path)
        
        print(f"Se encontraron {len(variables)} variables en total.")
        print("-" * 50)
        
        # Imprimimos las primeras 10 y las últimas 10 para ver cómo son
        print("Primeras 10 variables:")
        for name, shape in variables[:10]:
            print(f"  Nombre: {name:<50} | Forma: {shape}")
            
        print("\nÚltimas 10 variables:")
        for name, shape in variables[-10:]:
            print(f"  Nombre: {name:<50} | Forma: {shape}")

        print("-" * 50)

    except Exception as e:
        print(f"Error al leer el checkpoint: {e}")

if __name__ == '__main__':
    # TensorFlow necesita estar en modo Eager para esta función
    tf.enable_eager_execution()
    inspect_variables_in_checkpoint(CHECKPOINT_PATH)
