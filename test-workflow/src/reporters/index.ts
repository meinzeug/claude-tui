export { ConsoleReporter } from './ConsoleReporter';
export { JsonReporter, JsonReporterOptions } from './JsonReporter';
export { HtmlReporter, HtmlReporterOptions } from './HtmlReporter';
export { CoverageReporter, CoverageReporterOptions } from './CoverageReporter';

import { Reporter, TestConfig, ReporterConfig } from '../types';
import { ConsoleReporter } from './ConsoleReporter';
import { JsonReporter, JsonReporterOptions } from './JsonReporter';
import { HtmlReporter, HtmlReporterOptions } from './HtmlReporter';
import { CoverageReporter, CoverageReporterOptions } from './CoverageReporter';

export class ReporterFactory {
  static create(config: TestConfig, reporterConfig: ReporterConfig): Reporter {
    const { name, options = {} } = reporterConfig;

    switch (name.toLowerCase()) {
      case 'console':
        return new ConsoleReporter(config);
      
      case 'json':
        return new JsonReporter(config, options as JsonReporterOptions);
      
      case 'html':
        return new HtmlReporter(config, options as HtmlReporterOptions);
      
      case 'coverage':
        return new CoverageReporter(config, options as CoverageReporterOptions);
      
      default:
        throw new Error(`Unknown reporter: ${name}`);
    }
  }

  static createMultiple(config: TestConfig, reporterConfigs: ReporterConfig[]): Reporter[] {
    const reporters: Reporter[] = [];
    
    for (const reporterConfig of reporterConfigs) {
      try {
        reporters.push(this.create(config, reporterConfig));
      } catch (error) {
        console.warn(`⚠️  Failed to create reporter ${reporterConfig.name}: ${error}`);
      }
    }
    
    // Always add coverage reporter if coverage is enabled
    if (config.collectCoverage && !reporterConfigs.some(r => r.name === 'coverage')) {
      reporters.push(new CoverageReporter(config));
    }
    
    return reporters;
  }
}