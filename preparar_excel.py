"""Reconstruye muestras desde el Excel; no publica textos ni datos personales."""
from pathlib import Path
import argparse,hashlib,random,json
import openpyxl
import pandas as pd
from texto import TEMAS,limpiar,etiquetas_reglas,SEMILLA
P=Path(__file__).parent

def main(fuente):
 w=openpyxl.load_workbook(fuente,read_only=True,data_only=True);s=w.active
 rows=s.iter_rows(values_only=True);header=next(rows);required={'listing_id','review_id','comments',*TEMAS}
 if not required.issubset(header):raise ValueError('El Excel no tiene las columnas requeridas.')
 manual=[];total=0;fechas=[]
 for row in rows:
  d=dict(zip(header,row));total+=1
  if d.get('date'):fechas.append(str(d['date']))
  if all(d[t] in (0,1) for t in TEMAS):manual.append({k:d[k] for k in ['listing_id','review_id','comments']+TEMAS})
 m=pd.DataFrame(manual)
 if m.empty or m.comments.isna().any():raise ValueError('La muestra manual está vacía o tiene textos faltantes.')
 groups=set(m.listing_id);texts=set(m.comments.map(limpiar));seen=set();sample=[];rng=random.Random(SEMILLA);count=0
 exclusions={'alojamiento_manual':0,'texto_manual':0,'vacio_sin_letras':0,'duplicado_normalizado':0}
 for row in s.iter_rows(min_row=2,values_only=True):
  d=dict(zip(header,row))
  if d['listing_id'] in groups:exclusions['alojamiento_manual']+=1;continue
  clean=limpiar(d['comments'] or '')
  if clean in texts:exclusions['texto_manual']+=1;continue
  if not any(c.isalpha() for c in clean):exclusions['vacio_sin_letras']+=1;continue
  if clean in seen:exclusions['duplicado_normalizado']+=1;continue
  seen.add(clean);count+=1;item={k:d[k] for k in ['listing_id','review_id','comments']}
  if len(sample)<25000:sample.append(item)
  else:
   j=rng.randrange(count)
   if j<25000:sample[j]=item
 v=pd.DataFrame(sample);v[TEMAS]=[etiquetas_reglas(t) for t in v.comments]
 (P/'datos').mkdir(exist_ok=True);m.to_csv(P/'datos/muestra_manual.csv',index=False);v.to_csv(P/'datos/entrenamiento_debil.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 origen={'originales':total,'manuales':len(m),'entrenamiento':len(v),'elegibles':count,'exclusiones':exclusions,'fuente_archivo':fuente.name,'sha256_excel':hashlib.file_digest(open(fuente,'rb'),'sha256').hexdigest(),'ciudad_segun_notebook':'Santiago','fecha_dump':None,'fecha_primera_resena':min(fechas) if fechas else None,'fecha_ultima_resena':max(fechas) if fechas else None,'columnas_originales':list(header),'nota':'La fecha de reseña y la modificación del Excel no acreditan la fecha del dump. Los textos e identificadores de reseñadores no se publican.'}
 (P/'datos/origen.json').write_text(json.dumps(origen,ensure_ascii=False,indent=2));w.close();print('Datos preparados',origen,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--fuente',type=Path,default=P/'Etiquetados.xlsx');args=parser.parse_args()
 fuente=args.fuente if args.fuente.exists() else P.parent/'upload/Etiquetados.xlsx'
 main(fuente)
