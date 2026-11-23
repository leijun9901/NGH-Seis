"""Interface-anchored diagnostics for the frozen NGH-Seis v1.0 imaging controls.

Geological surfaces are defined before inspecting images. Envelope peaks are
explicit candidate events, never substitutes for geological interfaces. Failed
ties are retained with reasons. No correlation-window shift is used as a pick.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, hilbert

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.geology.blake_ew0008 import generate_blake_ew0008

DEPTH=(np.arange(512)-64)*3.
X=1310+np.arange(95)*37.5
FREQUENCY=14.

def first_crossing(field,threshold):
    top=np.full(field.shape[1],np.nan)
    for ix in range(field.shape[1]):
        indices=np.flatnonzero(field[:,ix]>=threshold)
        if len(indices):
            i=indices[0]
            if i and field[i,ix]!=field[i-1,ix]:
                fraction=(threshold-field[i-1,ix])/(field[i,ix]-field[i-1,ix])
                top[ix]=DEPTH[i-1]+3*fraction
            else:top[ix]=DEPTH[i]
    return top

def reconstruct(config,cid,seed,frozen):
    captured={}
    def capture(frame,event,arg):
        if event=='return' and frame.f_code.co_name=='generate_blake_ew0008':
            for key in ['structure_shift','depositional_tilt','x']:
                if key in frame.f_locals:
                    v=frame.f_locals[key]
                    captured[key]=v.copy() if isinstance(v,np.ndarray) else v
    sys.setprofile(capture)
    try:arrays,_=generate_blake_ew0008(config,cid,seed,include_proxy_image=False)
    finally:sys.setprofile(None)
    matches={k:bool(np.array_equal(arrays[k],frozen[k])) for k in
             ['Vp_full','rho_full','seafloor_m','bsr_m','lithology_proxy_full']}
    if not all(matches.values()):raise RuntimeError(f'Geology reconstruction differs for candidate {cid}: {matches}')
    offset=np.interp(X,captured['x'],captured['structure_shift'])+captured['depositional_tilt']*(X-captured['x'].mean())
    return offset,matches

def deep_surfaces(ai,offset):
    # Define horizons from the geology alone, not from either migration image.
    s=np.arange(650.,1251.,3.)
    profiles=np.stack([np.interp(s-offset[ix],DEPTH,np.log(np.maximum(ai[:,ix],1.))) for ix in range(95)],axis=1)
    profile=np.median(profiles,axis=1)
    derivative=np.gradient(gaussian_filter1d(profile,.7),3.)
    peaks,properties=find_peaks(np.abs(derivative),distance=40,prominence=1e-6)
    choices=[]
    for k in np.argsort(properties['prominences'])[::-1]:
        j=peaks[k];surface=s[j]-offset
        if 750<=np.median(surface)<=1120:
            choices.append((float(s[j]),surface,float(properties['prominences'][k])))
        if len(choices)==2:break
    return sorted(choices,key=lambda x:x[0])

def envelope(image):
    # Same conditioning and same envelope operator for released and control data.
    return gaussian_filter1d(np.abs(hilbert(image.astype(float),axis=0)),1.,axis=0)

def dominant_event(values,low,high,minimum_separation_m,dominance=1.5):
    indices=np.flatnonzero((DEPTH>=low)&(DEPTH<=high))
    if len(indices)<5:return None,'outside_imaged_support',[]
    start,end=indices[0],indices[-1]+1
    segment=values[start:end]
    # Compute prominence on the full trace. Truncating the wavelet to a narrow
    # geological search window would artificially suppress broad event peaks.
    all_peaks,_=find_peaks(values,distance=max(2,int(round(minimum_separation_m/3))),
                          prominence=.15*max(float(segment.max()),1e-30))
    peaks=all_peaks[(all_peaks>=start)&(all_peaks<end)]-start
    if not len(peaks):return None,'no_distinct_envelope_peak',[]
    order=np.argsort(segment[peaks])[::-1]
    candidates=(DEPTH[peaks[order]+start]).tolist()
    if len(order)>1 and segment[peaks[order[0]]] < dominance*segment[peaks[order[1]]]:
        return None,'competing_reflection_peaks',candidates
    # Peaks at a search edge are not acceptable evidence of an event location.
    j=int(peaks[order[0]]+start)
    if j-start<2 or end-1-j<2:return None,'peak_at_search_edge',candidates
    return float(DEPTH[j]),'candidate_peak',candidates

def describe(values):
    v=np.asarray(values,dtype=float)
    return dict(n=len(v),median=float(np.median(v)),p05=float(np.percentile(v,5)),p95=float(np.percentile(v,95))) if len(v) else dict(n=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True,help='Extracted complete release root')
    p.add_argument('--controls',type=Path,required=True,help='Five true-Vp constant-density control NPZs')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT/'configs/ngh_seis_geology.json')
    p.add_argument('--plot',action='store_true')
    p.add_argument('--kinematic-buffer-m',type=float,default=20.)
    p.add_argument('--peak-dominance',type=float,default=1.5)
    p.add_argument('--alignment-helper',type=Path,help='Optional directory containing the figure-authoring alignment QA helper')
    args=p.parse_args()
    if args.output.resolve().is_relative_to(args.release.resolve()):p.error('Output must be outside the frozen release')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];metadata=[];plot_data=[]
    for cid in range(1,6):
        with np.load(args.release/f'data/geology/sample_{cid:04d}_blake_geology.npz',allow_pickle=False) as f:
            g={k:f[k] for k in f.files}
        info=json.loads(str(g['metadata_json']))
        with np.load(args.release/f'data/training_pairs/sample_{cid:04d}_training_pair25.npz',allow_pickle=False) as f:
            released=f['input_rtm_conditioned_unscaled'][:,:95]
            migration=f['migration_Vp'][:,:95]
        with np.load(args.controls/f'sample_{cid:04d}_true_vp_constant_density.npz',allow_pickle=False) as f:
            reference=f['conditioned'][:,:95]
        offset,matches=reconstruct(args.config,cid,info['seed'],g)
        ai=g['AI_view'][:,:95].astype(float);vp=g['Vp_view'][:,:95].astype(float)
        sh=g['Sh_view'][:,:95];sg=g['Sg_view'][:,:95]
        bsr=np.interp(X,g['x_m'],g['bsr_m']-g['seafloor_m'])
        gas=first_crossing(sg,.003);support=first_crossing(sg,1e-6)
        gas_low=first_crossing(sg,.0015);gas_high=first_crossing(sg,.006)
        deep=deep_surfaces(ai,offset)
        horizons=[('BSR_geometric',bsr),('gas_top_Sg_0.003',gas)]
        horizons.extend((f'deep_depositional_{j+1}',surface) for j,(_,surface,_) in enumerate(deep))
        e_true,e_mig=envelope(reference),envelope(released)
        log_ai=np.log(np.maximum(ai,1.))
        for name,surface in horizons:
            for ix,z in enumerate(surface):
                row=dict(candidate_id=cid,family=info['style'],horizon=name,trace_index=ix,
                    geological_depth_m=float(z),event_true_vp_m=None,event_released_m=None,
                    true_vp_absolute_offset_m=None,released_absolute_offset_m=None,
                    event_displacement_m=None,status='',quarter_wavelength_m=None,
                    log_impedance_contrast_30m=None,true_peak_candidates='',released_peak_candidates='')
                row.update(vertical_time_proxy_depth_m=None,kinematic_gate_halfwidth_m=None)
                if not np.isfinite(z):
                    row['status']='no_gas_above_declared_threshold';rows.append(row);continue
                quarter=float(np.interp(z,DEPTH,vp[:,ix]))/(4*FREQUENCY)
                row['quarter_wavelength_m']=quarter
                contrast=float(np.interp(z+15,DEPTH,log_ai[:,ix])-np.interp(z-15,DEPTH,log_ai[:,ix]))
                row['log_impedance_contrast_30m']=contrast
                if name=='BSR_geometric' and (not np.isfinite(gas[ix]) or abs(gas[ix]-z)>quarter):
                    row['status']='BSR_reference_not_a_coincident_gas_contact';rows.append(row);continue
                if name.startswith('gas_top') and np.isfinite(support[ix]) and z-support[ix]>quarter:
                    row['status']='diffuse_threshold_crossing_not_a_unique_material_contact';rows.append(row);continue
                if abs(contrast)<.015:
                    row['status']='weak_local_impedance_contrast_below_1.5_percent';rows.append(row);continue
                true_pick,status,candidates=dominant_event(e_true[:,ix],z-quarter,z+quarter,quarter/2,args.peak_dominance)
                row['true_peak_candidates']=json.dumps(candidates)
                if true_pick is None:
                    row['status']='true_vp_'+status;rows.append(row);continue
                # Broad symmetric search allows shallower/deeper displacement;
                # neither a 1D velocity-mapping estimate nor correlation shifts
                # select the released-image peak.
                pick,status,candidates=dominant_event(e_mig[:,ix],z-180,z+180,quarter,args.peak_dominance)
                row['released_peak_candidates']=json.dumps(candidates)
                row['event_true_vp_m']=true_pick
                row['true_vp_absolute_offset_m']=true_pick-z
                if pick is None:
                    row['status']='released_'+status;rows.append(row);continue
                positive=DEPTH>=0
                zp=DEPTH[positive]
                st=1/vp[positive,ix];sm=1/migration[positive,ix]
                tt=np.r_[0.,np.cumsum((st[1:]+st[:-1])*1.5)]
                tm=np.r_[0.,np.cumsum((sm[1:]+sm[:-1])*1.5)]
                prediction=float(np.interp(np.interp(z,zp,tt),tm,zp))
                row.update(vertical_time_proxy_depth_m=prediction,
                           kinematic_gate_halfwidth_m=quarter+args.kinematic_buffer_m)
                # This is only a rejection gate on a peak already selected in
                # the broad symmetric image search. It does not replace the
                # observed peak or select a different peak near the prediction.
                if abs(pick-prediction)>quarter+args.kinematic_buffer_m:
                    row['status']='dominant_peak_inconsistent_with_vertical_time_gate';rows.append(row);continue
                row.update(event_released_m=pick,released_absolute_offset_m=pick-z,
                    event_displacement_m=pick-true_pick,status='provisional_interface_anchored_event')
                rows.append(row)
        meta=dict(candidate_id=cid,family=info['style'],geology_exact_matches=matches,
            coexistence_enabled=info.get('coexistence_enabled'),
            gas_top_minus_BSR_m=describe(gas[np.isfinite(gas)]-bsr[np.isfinite(gas)]),
            gas_threshold_0_0015_to_0_006_separation_m=describe((gas_high-gas_low)[np.isfinite(gas_high)&np.isfinite(gas_low)]),
            deep_reference_surfaces=[dict(depositional_coordinate_m=s,prominence_log_AI_per_m=prom) for s,_,prom in deep])
        metadata.append(meta);plot_data.append((cid,ai,reference,released,horizons))
    preliminary={(r['candidate_id'],r['horizon'],r['trace_index']):r for r in rows
                 if r['status']=='provisional_interface_anchored_event'}
    for row in rows:
        if row['status']!='provisional_interface_anchored_event':continue
        neighbours=[preliminary[(row['candidate_id'],row['horizon'],ix)]
                    for ix in range(max(0,row['trace_index']-4),min(95,row['trace_index']+5))
                    if (row['candidate_id'],row['horizon'],ix) in preliminary]
        if len(neighbours)<3:
            row['status']='insufficient_lateral_support';continue
        median=np.median([r['released_absolute_offset_m'] for r in neighbours])
        if abs(row['released_absolute_offset_m']-median)>15:
            row['status']='laterally_inconsistent_event_candidate'
    with (args.output/'interface_event_picks.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups={}
    for cid in range(1,6):
        for horizon in sorted({r['horizon'] for r in rows if r['candidate_id']==cid}):
            subset=[r for r in rows if r['candidate_id']==cid and r['horizon']==horizon]
            kept=[r for r in subset if r['status']=='provisional_interface_anchored_event']
            groups[f'{cid}_{horizon}']=dict(total_traces=len(subset),reasons=dict(Counter(r['status'] for r in subset)),
                provisional_released_offset_m=describe([r['released_absolute_offset_m'] for r in kept]),
                provisional_true_vp_offset_m=describe([r['true_vp_absolute_offset_m'] for r in kept]))
    summary=dict(definitions={
        'BSR_geometric':'Saved bsr_m minus seafloor_m; a prescribed reference surface, not necessarily a material discontinuity or observed reflector.',
        'gas_top_Sg_0.003':'Shallowest linearly interpolated Sg=0.003 crossing in each trace; absent traces remain absent. Operational saturation threshold, not automatically an acoustic boundary.',
        'deep_depositional':'Constant depositional-coordinate surfaces selected from the two strongest, separated median log-AI derivative peaks in the deep sediment. Selected from geology before image inspection; the stochastic generator does not define sharp layer boundaries.'},
        gates=dict(contrast_30m_log_AI=.015,true_image_search='geological surface +/- one quarter wavelength',
            released_search_m=[-180,180],envelope_peak_dominance=args.peak_dominance,source_peak_frequency_Hz=14,
            kinematic_rejection='A previously selected image peak must be within one quarter wavelength plus the declared buffer of an independent 1D vertical-time proxy; the proxy is not reported as an observed event location.',
            kinematic_buffer_m=args.kinematic_buffer_m,
            lateral_gate='At least 3 preliminary candidates in +/-4 traces; offset within 15 m of their median; one noniterative pass'),
        interpretation='Provisional associations are envelope-energy event candidates referenced to defined geological surfaces, not proven unique boundary picks. Absolute offsets include wavelet/gradient point-spread offset. Failed associations are not assigned correlation-window shifts.',
        uncertainty='Quarter-wavelength and 3 m sampling; no independence assumption or inferential confidence intervals for lateral traces.',
        metadata=metadata,groups=groups)
    (args.output/'interface_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if args.plot:plot_figures(plot_data,rows,args.output,args.alignment_helper)
    print(json.dumps({k:v['reasons'] for k,v in groups.items()},indent=2))

def plot_figures(plot_data,rows,out,alignment_helper=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':9,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors=['#357F8A','#CB8952','#7862A5','#5674A5']
    for cid,ai,true,mig,horizons in plot_data:
        fig,axes=plt.subplots(1,3,figsize=(7.2,4.5),sharex=True,sharey=True)
        fig.subplots_adjust(left=.085,right=.985,bottom=.32,top=.89,wspace=.12)
        visible=(DEPTH>=150)&(DEPTH<=1250)
        scale=np.percentile(np.abs(np.concatenate([true[visible].ravel(),mig[visible].ravel()])),99.5)
        contrast=np.abs(np.gradient(np.log(np.maximum(ai,1.)),3.,axis=0))
        for ax,image,title,cmap,lim in zip(axes,[contrast,true/scale,mig/scale],
                ['True impedance transitions','True-Vp gradient','Released gradient'],['cividis','gray','gray'],[(0.,.006),(-1,1),(-1,1)]):
            ax.imshow(image[visible],extent=(-.01875,3.54375,(DEPTH[visible][-1]+1.5)/1000,(DEPTH[visible][0]-1.5)/1000),
                cmap=cmap,vmin=lim[0],vmax=lim[1],aspect='auto',interpolation='nearest')
            ax.set(title=title,xlabel='Distance (km)',ylim=(1.25,.15),xticks=[0,1,2,3])
            for (name,z),color in zip(horizons,colors):
                zplot=np.where((z>=150)&(z<=1250),z/1000,np.nan)
                ax.plot(np.arange(95)*.0375,zplot,color=color,lw=.7,ls='--')
        for row in rows:
            if row['candidate_id']==cid and row['status']=='provisional_interface_anchored_event':
                j=[h[0] for h in horizons].index(row['horizon']);color=colors[j]
                if .15<=row['event_true_vp_m']/1000<=1.25:
                    axes[1].plot(row['trace_index']*.0375,row['event_true_vp_m']/1000,'.',color=color,ms=2)
                if .15<=row['event_released_m']/1000<=1.25:
                    axes[2].plot(row['trace_index']*.0375,row['event_released_m']/1000,'.',color=color,ms=2)
        axes[0].set_ylabel('Depth below seafloor (km)')
        for label,ax in zip('abc',axes):ax.text(-.10,1.07,label,transform=ax.transAxes,weight='bold',fontsize=11)
        labels=['Geometric BSR','Gas-top threshold','Deep surface 1','Deep surface 2']
        for (_,z),color,label in zip(horizons,colors,labels):axes[0].plot([],[],ls='--',color=color,label=label)
        cax=fig.add_axes([.11,.19,.23,.016])
        cb=fig.colorbar(axes[0].images[0],cax=cax,orientation='horizontal',ticks=[0,.003,.006])
        cb.set_label(r'$|\partial_z\ln AI|$ (m$^{-1}$)',fontsize=10)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.62,.025))
        fig.canvas.draw()
        rectangles=[ax.get_position().bounds for ax in axes]
        assert max(r[2] for r in rectangles)-min(r[2] for r in rectangles)<1e-6
        assert max(r[3] for r in rectangles)-min(r[3] for r in rectangles)<1e-6
        if alignment_helper:
            sys.path.insert(0,str(alignment_helper))
            from audit_panel_alignment import require_matplotlib_panel_alignment
            require_matplotlib_panel_alignment(fig,axes=list(axes),
                json_out=out/f'candidate_{cid:04d}.alignment.json',strict=True)
        fig.savefig(out/f'candidate_{cid:04d}_interface_comparison.pdf')
        fig.savefig(out/f'candidate_{cid:04d}_interface_comparison.svg')
        fig.savefig(out/f'candidate_{cid:04d}_interface_comparison.png',dpi=600)
        plt.close(fig)

if __name__=='__main__':main()
