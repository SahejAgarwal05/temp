#!/usr/bin/env python3
"""Benchmark full render time and select only configurations passing every checker run."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import shutil
import subprocess


def positive_list(value):
    values = [float(item) for item in value.split(',')]
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise argparse.ArgumentTypeError('use comma-separated finite positive distances')
    return values


def run(executable, scene, reference, output, options):
    command = [executable, scene, '--output', str(output), '--correctness-check', reference, *options]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time = re.search(r'Time taken: ([\d.eE+\-]+) seconds', result.stdout)
    error = re.search(r'^Error: ([\d.eE+\-]+)', result.stdout, re.MULTILINE)
    passed = result.returncode == 0 and 'Correctness check: PASS' in result.stdout
    output.with_suffix('.log').write_text('$ ' + ' '.join(command) + '\n' + result.stdout)
    seconds = float(time[1]) if time else None
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        seconds = None
    return seconds, float(error[1]) if error else None, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='./path_tracing')
    parser.add_argument('--case', nargs=2, action='append', required=True,
                        metavar=('SCENE', 'REFERENCE'), help='repeat for every scene that must pass')
    parser.add_argument('--spacing', type=positive_list, default=positive_list('.01,.02,.04'))
    parser.add_argument('--radius', type=positive_list, default=positive_list('.01,.02,.04'))
    parser.add_argument('--cube-size', type=positive_list, default=positive_list('.04,.08,.16'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, default=Path('grid_benchmark'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {}
    best = None
    csv_path = args.output_dir / 'results.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'backend', 'D', 'R', 'cube_size', 'case', 'repeat', 'seconds', 'error', 'pass', 'speedup'])
        writer.writeheader()

        def measure(backend, options, distances=('', '', ''), executable=None):
            medians = []
            passes = True
            for case, (scene, reference) in enumerate(args.case):
                times = []
                for repeat in range(args.repeats):
                    prefix = f'{backend}_{distances[0]}_{distances[1]}_{distances[2]}_{case}_{repeat}'
                    seconds, error, passed = run(executable or args.executable, scene, reference,
                        args.output_dir / (prefix + '.ppm'),
                        ['--intersection', backend, '--seed', str(args.seed), *options])
                    passes &= passed and seconds is not None
                    if seconds is not None:
                        times.append(seconds)
                    speedup = baseline[case] / seconds if case in baseline and seconds else ''
                    row = dict(zip(('D', 'R', 'cube_size'), distances))
                    row.update(backend=backend, case=scene, repeat=repeat,
                               seconds=seconds, error=error, **{'pass': passed}, speedup=speedup)
                    writer.writerow(row)
                    stream.flush()
                    print(f'{prefix}: {seconds}s, error={error}, {"PASS" if passed else "FAIL"}', flush=True)
                medians.append(statistics.median(times) if times else math.inf)
            return passes, medians

        passed, times = measure('bvh', [])
        if not passed:
            raise SystemExit(f'BVH baseline failed a checker/run; inspect {args.output_dir} logs. '
                             'References must include max-error metadata.')
        baseline.update(enumerate(times))
        project = Path(__file__).resolve().parents[1]
        for spacing, cube in itertools.product(args.spacing, args.cube_size):
            definitions = f'-DSAMPLED_SPACING={spacing}f -DSAMPLED_CUBE_SIZE={cube}f'
            build = subprocess.run(['make', '-B', 'path_tracing', f'CPPFLAGS={definitions}'],
                                   cwd=project, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (args.output_dir / f'build_{spacing}_{cube}.log').write_text(build.stdout)
            if build.returncode:
                raise SystemExit(f'Build failed for D={spacing}, cube={cube}; inspect build log')
            binary = (args.output_dir / f'renderer_{spacing}_{cube}').resolve()
            shutil.copy2(project / 'path_tracing', binary)
            for radius in args.radius:
                options = ['--acceptance-radius', str(radius)]
                passed, times = measure('sampled', options, (spacing, radius, cube), str(binary))
                total = sum(times)
                if passed and (best is None or total < best['total_median_seconds']):
                    best = dict(D=spacing, R=radius, cube_size=cube,
                                total_median_seconds=total, speedup=sum(baseline.values()) / total,
                                case_median_seconds=times, options=options,
                                build_definitions=definitions, executable=str(binary))
    summary = dict(cases=args.case, repeats=args.repeats, seed=args.seed,
                   baseline_median_seconds=list(baseline.values()), best=best)
    (args.output_dir / 'best.json').write_text(json.dumps(summary, indent=2) + '\n')
    if best is None:
        raise SystemExit(f'No sampled settings passed all checker runs. Results: {csv_path}')
    print(f'Fastest passing constants: D={best["D"]}, cube_size={best["cube_size"]}')
    print(f'Run: {best["executable"]} SCENE {" ".join(best["options"])}')
    print(f'Speedup against BVH: {best["speedup"]:.3f}x; results: {csv_path}')


if __name__ == '__main__':
    main()
